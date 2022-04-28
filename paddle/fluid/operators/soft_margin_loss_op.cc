/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SoftMarginLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SoftMarginLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SoftMarginLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "SoftMarginLossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SoftMarginLossGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "SoftMarginLossGrad");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(x_dims, labels_dims,
                        platform::errors::InvalidArgument(
                            "Input(X) and Input(Label) shall have the same "
                            "shape. But received: the shape of Input(X) is "
                            "[%s], the shape of Input(Label) is [%s].",
                            x_dims, labels_dims));

      PADDLE_ENFORCE_EQ(x_dims, dout_dims,
                        platform::errors::InvalidArgument(
                            "Input(X) and Input(Out@Grad) shall have the same "
                            "shape. But received: the shape of Input(X) is "
                            "[%s], the shape of Input(Out@Grad) is [%s].",
                            x_dims, dout_dims));
    }

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SoftMarginLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), the input is a tensor of logits"
             "computed by the previous operator. ");
    AddInput("Label",
             "(Tensor, default Tensor<float>), have same shape with input"
             "label should between in 0 and 1.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), have same shape with"
              "input");
    AddComment(R"DOC(
SoftMarginLoss operator.
This measures the element-wise probability error in classification tasks
in which each class is independent.
The logitstic loss is given as follows:
      $$loss = log(1+exp(-Label * X))$$
)DOC");
  }
};

template <typename T>
class SoftMarginLossGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("soft_margin_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(SoftMarginLossInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(SoftMarginLossGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(soft_margin_loss,
                            SoftMarginLossInferShapeFunctor,
                            PD_INFER_META(phi::SoftMarginLossInferMeta));

REGISTER_OPERATOR(soft_margin_loss, ops::SoftMarginLossOp,
                  ops::SoftMarginLossOpMaker,
                  ops::SoftMarginLossGradOpMaker<paddle::framework::OpDesc>,
                  ops::SoftMarginLossGradOpMaker<paddle::imperative::OpBase>,
                  ops::SoftMarginLossInplaceInferer, SoftMarginLossInferShapeFunctor);
REGISTER_OPERATOR(soft_margin_loss_grad, ops::SoftMarginLossGradOp,
                  ops::SoftMarginLossGradInplaceInferer);
