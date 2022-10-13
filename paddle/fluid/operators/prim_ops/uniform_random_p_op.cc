// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class UniformRandomPrimOp : public framework::OperatorBase {
 public:
  UniformRandomPrimOp(const std::string &type,
                      const framework::VariableNameMap &inputs,
                      const framework::VariableNameMap &outputs,
                      const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator uniform_randrom_p should not be excuted directly"));
  }
};

class UniformRandomPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("ShapeTensor",
             "(Tensor<int64_t> or Tensor<int32_t>, optional) . If provided, "
             "uniform_random_p "
             "according to "
             "this given shape. It means that it has a higher priority than "
             "the shape attribute, while the shape attribute still should be "
             "set correctly to guarantee shape inference in compile time.")
        .AsDispensable();
    AddInput("ShapeTensorList",
             "(vector<Tensor<int64_t>> or vector<Tensor<int32_t>>, optional). "
             "If provided, uniform_random use this. The shape of the tensor "
             "must be [1], it has the highest priority comparing with "
             "Input(ShapeTensor) and attr(shape).")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "(Tensor), The output tensor of uniform_random_p op.");
    AddAttr<std::vector<int64_t>>("shape", "The shape of the output tensor")
        .SetDefault({});
    AddAttr<float>("min", "Minimum value of uniform_random_p. [default -1.0].");
    AddAttr<float>("max", "Maximun value of uniform_random_p. [default 1.0].");
    AddAttr<int>("seed",
                 "Random seed used for generating samples. "
                 "0 means use a seed generated by the system."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time. ");
    AddAttr<int>("dtype", "Output tensor data type. ");
    AddComment(R"DOC(
Autograd primitive uniform_random_p operator.
)DOC");
  }
};

class UniformRandomPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "UniformRandomPOp");

    PADDLE_ENFORCE_LT(
        ctx->Attrs().Get<float>("min"),
        ctx->Attrs().Get<float>("max"),
        platform::errors::InvalidArgument(
            "The uniform_random_p's min must less then max. But received min = "
            "%f great than or equal max = %f.",
            ctx->Attrs().Get<float>("min"),
            ctx->Attrs().Get<float>("max")));

    if (ctx->HasInputs("ShapeTensorList")) {
      // top prority shape
      auto inputs_name = ctx->Inputs("ShapeTensorList");
      PADDLE_ENFORCE_GT(inputs_name.size(),
                        0,
                        platform::errors::InvalidArgument(
                            "Input(ShapeTensorList)'size of "
                            "Op(uniform_random_p) can't be zero."
                            "Please check the Attr(shape)'s size of"
                            "Op(uniform_random_p).)"));
      auto out_dims = std::vector<int>(inputs_name.size(), -1);
      ctx->SetOutputDim("Out", phi::make_ddim(out_dims));
      return;
    }
    auto &shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    if (ctx->HasInput("ShapeTensor") && shape.empty()) {
      auto shape_dims = ctx->GetInputDim("ShapeTensor");
      PADDLE_ENFORCE_EQ(
          shape_dims.size(),
          1,
          platform::errors::InvalidArgument(
              "ShapeError: Input(ShapeTensor)' dimension size of "
              "Op(uniform_random_p) must be 1."
              "But received ShapeTensor's dimensions = %d, shape = [%s]",
              shape_dims.size(),
              shape_dims));
      int num_ele = 1;
      for (int i = 0; i < shape_dims.size(); ++i) {
        num_ele *= shape_dims[i];
      }
      auto vec_dims = std::vector<int64_t>(num_ele, -1);
      auto out_dims = phi::make_ddim(vec_dims);
      ctx->SetOutputDim("Out", out_dims);
      return;
    }

    PADDLE_ENFORCE_EQ(shape.empty(),
                      false,
                      platform::errors::InvalidArgument(
                          "if there is no Input(ShapeTensorList) and no "
                          "Input(ShapeTensor),the "
                          "attr(shape) information must "
                          "be set by Attr(shape)."));
    std::vector<int64_t> tensor_shape;
    tensor_shape.reserve(shape.size());
    for (auto dim : shape) {
      tensor_shape.push_back(static_cast<int64_t>(dim));
    }
    ctx->SetOutputDim("Out", phi::make_ddim(tensor_shape));
  }
};

class UniformRandomPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto y_name = Output(ctx, "Out")[0];
    auto data_type = static_cast<framework::proto::VarType::Type>(
        PADDLE_GET_CONST(int, ctx->GetAttr("dtype")));
    SetDataType(ctx, y_name, data_type);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(uniform_random_p,
                  paddle::operators::UniformRandomPrimOp,
                  paddle::operators::UniformRandomPrimOpMaker,
                  paddle::operators::UniformRandomPrimOpShapeInference,
                  paddle::operators::UniformRandomPrimOpVarTypeInference);
