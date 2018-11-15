// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/tree_conv_op.h"

namespace paddle {
namespace operators {
class TreeConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("NodesVector",
             "(Tensor) The feature vector of every node on the tree"
             "The shape of the feature vector must be "
             "[max_tree_node_size, feature_size]");
    AddInput("EdgeSet",
             "(Tensor) The Edges of Tree. the edge must be directional"
             "The shape of the edge set must be [max_tree_node_size, 2]");
    AddInput("Filter",
             "(Tensor) The feature detector"
             "The shape of the filter is "
             "[feature_size, 3, output_size, num_filters]");
    AddOutput(
        "Out",
        "(Tensor) The feature vector of subtrees"
        "The shape of the output tensor is [max_tree_node_size, output_size]"
        "The output tensor could be a new feature "
        "vector for next tree convolution layers");
    AddAttr<int>("max_depth", "(int, default: 2) The depth of feature detector")
        .SetDefault(2)
        .GreaterThan(1);
    AddComment(R"DOC(
Tree-Based Convolution Operator.
Tree-Based Convolution is a kind of convolution based on tree structure.
Tree-Based Convolution is a part of Tree-Based Convolution Neural Network(TBCNN),
which is used to classify tree structures, such as Abstract Syntax Tree.
The paper of Tree-Based Convolution Operator is here:
https://arxiv.org/abs/1409.5718v1
)DOC");
  }
};
class TreeConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"));
    auto edge_dims = ctx->GetInputDim("EdgeSet");
    auto vector_dims = ctx->GetInputDim("NodesVector");
    auto filter_dims = ctx->GetInputDim("Filter");
    PADDLE_ENFORCE_EQ(
        edge_dims[2], 2,
        "The shape of EdgeSet Tensor is [batch_size, max_tree_node_size, 2]");
    PADDLE_ENFORCE_EQ(edge_dims.size(), 3,
                      "The dimension of EdgeSet Tensor is 3");
    PADDLE_ENFORCE_EQ(vector_dims.size(), 3, "The dimension of Input Tensor");
    PADDLE_ENFORCE_EQ(filter_dims.size(), 4,
                      "The dimension of Filter Tensor is 4");
    PADDLE_ENFORCE_EQ(filter_dims[1], 3,
                      "The shape of Filter Tensor is "
                      "[feature_size, 3, output_size, num_filters]");
    PADDLE_ENFORCE_EQ(filter_dims[0], vector_dims[2],
                      "filter_dims[0] must equal to vector_dims[2]");
    auto output_dims = framework::make_ddim(
        {vector_dims[0], vector_dims[1], filter_dims[2], filter_dims[3]});
    ctx->SetOutputDim("Out", output_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("NodesVector")->type()),
        ctx.device_context());
  }
};

class TreeConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto vectors_dims = ctx->GetInputDim("NodesVector");
    auto filter_dims = ctx->GetInputDim("Filter");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "the gradient of output(Out) must not be null");
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("NodesVector"))) {
      ctx->SetOutputDim(framework::GradVarName("NodesVector"), vectors_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("NodesVector")->type()),
        ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(tree_conv, ops::TreeConvOp, ops::TreeConvOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OPERATOR(tree_conv_grad, ops::TreeConvGradOp);

REGISTER_OP_CPU_KERNEL(
    tree_conv, ops::TreeConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TreeConvKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    tree_conv_grad,
    ops::TreeConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TreeConvGradKernel<paddle::platform::CPUDeviceContext, double>);
