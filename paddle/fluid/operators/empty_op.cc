/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/empty_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class EmptyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(empty operator
Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument shape.

The type of the tensor is specify by `dtype`.
)DOC");
    AddOutput("Out", "(LoDTensor) The output tensor.");
    AddAttr<std::vector<int>>("shape", "The shape of output tensor");
    AddAttr<int>("dtype", "The data type of output tensor, Default is float")
        .SetDefault(framework::proto::VarType::FP32);
    // AddAttr<bool>("force_cpu",
    //               "Whether the output tensor must be at CPU memory or not. "
    //               "Default is false.")
    //     .SetDefault(false);
  }
};

class EmptyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "empty");
    auto& shape = context->Attrs().Get<std::vector<int>>("shape");
    context->SetOutputDim("Out", framework::make_ddim(shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class EmptyOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, ctx->GetAttr("dtype")));
    ctx->SetOutputDataType("Out", data_type);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    empty, ops::EmptyOp, ops::EmptyOpMaker, ops::EmptyOpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(empty, ops::EmptyKernel<plat::CPUDeviceContext, float>,
                       ops::EmptyKernel<plat::CPUDeviceContext, double>,
                       ops::EmptyKernel<plat::CPUDeviceContext, int64_t>,
                       ops::EmptyKernel<plat::CPUDeviceContext, int>,
                       ops::EmptyKernel<plat::CPUDeviceContext, plat::float16>);
