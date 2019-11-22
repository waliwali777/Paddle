/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dropout_with_seed_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

using framework::Tensor;

class DropoutWithSeedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("Mask", x_dims);
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class DropoutWithSeedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of dropout_with_seed op.");
    AddInput("Seed", "The seed of dropout_with_seed op.");
    AddOutput("Out", "The output of dropout_with_seed op.");
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();

    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float& drop_p) {
          PADDLE_ENFORCE(drop_p >= 0.0f && drop_p <= 1.0f,
                         "'dropout_prob' must be between 0.0 and 1.0.");
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_prob)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_prob)"
        "2. upscale_in_train, upscale the outcome at training time, do nothing "
        "in inference"
        "   train: out = input * mask / ( 1.0 - dropout_prob )"
        "   inference: out = input"
        "   dropout op can be removed from the program. the program will be "
        "efficient")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string& type) {
          PADDLE_ENFORCE(
              type == "downgrade_in_infer" || type == "upscale_in_train",
              "dropout_implementation can only be downgrade_in_infer or "
              "upscale_in_train");
        });

    AddComment(R"DOC(
DropoutWithSeed Operator.

DropoutWithSeed refers to randomly dropping out units in a nerual network. It is a
regularization technique for reducing overfitting by preventing neuron
co-adaption during training. The dropout operator randomly set (according to
the given dropout probability) the outputs of some units to zero, while others
are set equal to their corresponding inputs.

)DOC");
  }
};

class DropoutWithSeedOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"), false,
                      "GradOp is only callable when is_test is false");

    PADDLE_ENFORCE(ctx->HasInput("Mask"), "Mask must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) must not be null.");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->ShareLoD(framework::GradVarName("Out"),
                  /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
    // return framework::OpKernelType(
    //    ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
    //    ctx.GetPlace());
  }
};

template <typename T>
class DropoutWithSeedGradOpMaker : public framework::SingleGradOpMaker {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("dropout_with_seed_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Mask", this->Output("Mask"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dropout_with_seed, ops::DropoutWithSeedOp,
                  ops::DropoutWithSeedOpMaker,
                  ops::DropoutWithSeedGradOpMaker<paddle::framework::OpDesc>,
                  ops::DropoutWithSeedGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(dropout_with_seed_grad, ops::DropoutWithSeedOpGrad);
REGISTER_OP_CPU_KERNEL(
    dropout_with_seed,
    ops::CPUDropoutWithSeedKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUDropoutWithSeedKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    dropout_with_seed_grad,
    ops::DropoutWithSeedGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DropoutWithSeedGradKernel<paddle::platform::CPUDeviceContext, double>);
