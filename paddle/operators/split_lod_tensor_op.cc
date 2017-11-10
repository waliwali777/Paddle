/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/op_registry.h"
#include "paddle/memory/memcpy.h"

namespace paddle {
namespace operators {

struct CopyRange {
  size_t begin;
  size_t end;
};

using LoD = framework::LoD;

class SplitLoDTensorOp : public framework::OperatorBase {
 public:
  SplitLoDTensorOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto &mask = scope.FindVar(Input("Mask"))->Get<framework::LoDTensor>();
    auto *out_true =
        scope.FindVar(Output("OutTrue"))->GetMutable<framework::LoDTensor>();
    auto *out_false =
        scope.FindVar(Output("OutFalse"))->GetMutable<framework::LoDTensor>();
    auto level = static_cast<size_t>(Attr<int>("level"));

    auto &x_lod = x.lod();
    //    auto x_lod_level = x_lod.size();

    auto &mask_dim = mask.dims();
    auto *mask_data = mask.data<bool>();

    // TODO(qijun): handle some corner case
    // if (x_lod_level == 0) {
    //   // no lod
    //   PADDLE_ENFORCE_EQ(x_dim[0], mask_dim.size());

    // } else {
    //   PADDLE_ENFORCE_LT(level, x_lod_level);
    //   cur_level_lod = x_lod_level[level];
    //   PADDLE_ENFORCE_EQ(cur_level_lod.size() - 1, mask_dim.size());

    // }

    std::vector<std::vector<CopyRange>> copy_ranges(mask_dim[0]);

    // set out_true/out_false lod
    for (size_t t = 0; t < 2; t++) {
      LoD *lod = nullptr;
      if (t == 0) {
        lod = out_false->mutable_lod();
      } else {
        lod = out_true->mutable_lod();
      }
      lod->clear();
      for (size_t i = 0; i < static_cast<size_t>(mask_dim[0]); i++) {
        if (static_cast<size_t>(mask_data[i]) == t) {
          size_t start_idx = i;
          auto lod_and_offset = framework::GetSubLoDAndAbsoluteOffset(
              x_lod, start_idx, start_idx + 1, level);

          auto &lod_length = lod_and_offset.first;
          framework::AppendLoD(lod, lod_length);

          size_t start_offset = lod_and_offset.second.first;
          size_t end_offset = lod_and_offset.second.second;
          copy_ranges[t].emplace_back(CopyRange{start_offset, end_offset});
        }
      }
    }

    for (size_t t = 0; t < 2; ++t) {
      framework::LoDTensor *out;
      if (t == 0) {
        out = out_false;
      } else {
        out = out_true;
      }
      auto &ranges = copy_ranges[t];
      size_t height = std::accumulate(
          ranges.begin(), ranges.end(), 0UL,
          [](size_t a, const CopyRange &b) { return a + b.end - b.begin; });
      auto x_dim = x.dims();
      x_dim[0] = static_cast<int64_t>(height);
      out->Resize(x_dim);
      out->mutable_data(x.place(), x.type());
      size_t offset = 0;
      for (auto &each_range : ranges) {
        size_t len = each_range.end - each_range.begin;
        if (len == 0) {
          continue;
        }
        // out[offset: offset+len] = x[each_range.begin: each_range.end]
        out->Slice(static_cast<int>(offset), static_cast<int>(offset + len))
            .CopyFrom(x.Slice(static_cast<int>(each_range.begin),
                              static_cast<int>(each_range.end)),
                      x.place(), dev_ctx);
        offset += len;
      }
    }
  }
};

class SplitLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SplitLoDTensorOpProtoMaker(framework::OpProto *proto,
                             framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("Mask", "");
    AddOutput("OutTrue", "");
    AddOutput("OutFalse", "");
    AddAttr<int>("level", "(int) the specific lod level to rank.")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(
        R"DOC()DOC");
  }
};

class SplitLoDTensorInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "SplitLoDTensorOp must has input X.");
    PADDLE_ENFORCE(context->HasInput("Mask"),
                   "SplitLoDTensorOp must has input Mask.");
    PADDLE_ENFORCE(context->HasOutput("OutTrue"),
                   "SplitLoDTensorOp must has output OutTrue.");
    PADDLE_ENFORCE(context->HasOutput("OutFalse"),
                   "SplitLoDTensorOp must has output OutFalse.");

    auto mask_dim = context->GetInputDim("Mask");
    PADDLE_ENFORCE_EQ(mask_dim.size(), 2);
    PADDLE_ENFORCE_EQ(mask_dim[1], 1);

    context->SetOutputDim("OutTrue", context->GetInputDim("X"));
    context->SetOutputDim("OutFalse", context->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(split_lod_tensor, ops::SplitLoDTensorOp,
                  ops::SplitLoDTensorOpProtoMaker,
                  ops::SplitLoDTensorInferShape,
                  paddle::framework::EmptyGradOpMaker);
