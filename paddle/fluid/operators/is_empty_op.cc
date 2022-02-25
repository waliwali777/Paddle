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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class IsEmptyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), x->place());
  }
};

class IsEmptyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) Tensor which is to be checked.");
    AddOutput("Out",
              "(LoDTensor) a boolean Tensor that indicate empty or not.");
    AddComment(R"DOC(
IsEmpty Operator which checks whether a tensor is empty.

It will just return product(tensor.ddims()) > 0;
              )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DELCARE_INFER_SHAPE_FUNCTOR(is_empty, IsEmptyInferShapeFunctor,
                            PT_INFER_META(phi::IsEmptyInferMeta));
REGISTER_OPERATOR(
    is_empty, ops::IsEmptyOp, ops::IsEmptyOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    IsEmptyInferShapeFunctor);

