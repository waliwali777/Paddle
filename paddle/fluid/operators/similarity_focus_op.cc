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

#include "paddle/fluid/operators/similarity_focus_op.h"

namespace paddle {
namespace operators {
class SimilarityFocusOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 4-D tensor with shape,"
             " [BatchSize, X, Y, Z]");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), the similarity focus mask"
              " with the same shape of input X.");
    AddAttr<int>("axis",
                 "(int32), indicating the dimension to be select. It can"
                 " only be 1, 2, or 3.");
    AddAttr<std::vector<int>>("indexes",
                              "(std::vector<int32>), indicating the indexes"
                              " of the selected dimension.");
    AddComment(R"DOC(
SimilarityFocus Operator.

Generate a similarity focus mask with the same shape of input using the following method:
1. Extract the 4-D matrix(here the first dimension is BatchSize) corresponding
   to the axis according to the indexes. For example, if axis=1 and indexes=[a],
   it will get the matrix T=X[:, a, :, :]. In this case, if the shape of input X
   is (BatchSize, A, B, C), the shape of matrix T is (BatchSize, B, C).
2. For each index, find the largest numbers in the matrix T, so that the same
   row and same column has at most one number(obviously there will be min(B, C)
   numbers), and mark the corresponding position of the 3-D similarity focus mask
   as 1, otherwise as 0. Do elementwise-or for each index.
3. Broadcast the 3-D similarity focus mask to the same shape of input X.

Refer to `Similarity Focus Layer <http://www.aclweb.org/anthology/N16-1108>`_
)DOC");
  }
};

class SimilarityFocusOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 4, "Input(X)'s rank should be 4.");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        platform::CPUPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(similarity_focus, ops::SimilarityFocusOp,
                  ops::SimilarityFocusOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(similarity_focus, ops::SimilarityFocusKernel<float>,
                       ops::SimilarityFocusKernel<double>);
