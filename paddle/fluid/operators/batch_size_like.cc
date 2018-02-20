/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/batch_size_like.h"

namespace paddle {
namespace operators {

void BatchSizeLikeOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "Input(Input) of %s should not be null.", Type());
  PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) of %s should not be null.",
                 Type());

  auto &shape = ctx->Attrs().Get<std::vector<int>>("shape");
  PADDLE_ENFORCE_GT(shape.size(), 0);
  std::vector<int64_t> shape_int64(shape.size(), 0);
  std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                 [](int a) { return static_cast<int64_t>(a); });
  auto output_dim = framework::make_ddim(shape_int64);

  int input_dim_idx = ctx->Attrs().Get<int>("input_dim_idx");
  PADDLE_ENFORCE_GE(input_dim_idx, 0);
  PADDLE_ENFORCE_GT(ctx->GetInputDim("Input").size(), input_dim_idx);

  int output_dim_idx = ctx->Attrs().Get<int>("output_dim_idx");
  PADDLE_ENFORCE_GE(output_dim_idx, 0);
  PADDLE_ENFORCE_GT(static_cast<int>(shape.size()), output_dim_idx);

  output_dim[output_dim_idx] = ctx->GetInputDim("Input")[input_dim_idx];
  ctx->SetOutputDim("Out", output_dim);
}

BatchSizeLikeOpMaker::BatchSizeLikeOpMaker(OpProto *proto,
                                           OpAttrChecker *op_checker)
    : framework::OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput("Input",
           "(Tensor) Tensor "
           "whose input_dim_idx'th dimension specifies the batch_size");
  AddOutput("Out",
            "(Tensor) Tensor of specified shape will be filled "
            "with the specified value");
  AddAttr<std::vector<int>>("shape", "(vector<int>) The shape of the output");
  AddAttr<int>("input_dim_idx",
               "(int, default 0) The index of input's batch size dimension")
      .SetDefault(0);
  AddAttr<int>("output_dim_idx",
               "(int, default 0) The index of output's batch size dimension")
      .SetDefault(0);
}

}  // namespace operators
}  // namespace paddle
