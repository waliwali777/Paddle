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

#include "paddle/operators/gather_op.h"
#include "paddle/framework/ddim.h"

namespace paddle {
namespace operators {

class GatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of GatherOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Index"),
                   "Input(Index) of GatherOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GatherOp should not be null.");

    int batch_size = ctx->GetInputDim("Index")[0];
    PADDLE_ENFORCE_GE(batch_size, 0, "Batch size must be >0");
    framework::DDim output_dims(ctx->GetInputDim("X"));
    output_dims[0] = batch_size;
    ctx->SetOutputDim("Out", output_dims);
  }
};

class GatherGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class GatherOpMaker : public framework::OpInfoMaker {
 public:
  GatherOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpInfoMaker(proto, op_checker) {
    AddInput("X", "The source input of gather op");
    AddInput("Index", "The index input of gather op");
    AddOutput("Out", "The output of add op");
    AddComment(R"DOC(
Gather Operator by selecting from the first axis,

Out = X[Index]
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(gather, ops::GatherOp, ops::GatherOpMaker, gather_grad,
            ops::GatherGradOp);
REGISTER_OP_CPU_KERNEL(gather,
                       ops::GatherOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    gather_grad,
    ops::GatherGradientOpKernel<paddle::platform::CPUPlace, float>);
