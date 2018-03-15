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

#include "paddle/fluid/operators/sequence_expand_op.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;

class SequenceExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceExpandOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of SequenceExpandOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceExpandOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2U,
                      "Dimension number of Input(X) should be 2.");
    int ref_level = ctx->Attrs().Get<int>("ref_level");

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
      framework::Variable* y_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("Y")[0]);

      auto& x_lod = x_var->Get<LoDTensor>().lod();
      auto& y_lod = y_var->Get<LoDTensor>().lod();

      PADDLE_ENFORCE_LE(x_lod.size(), 1,
                        "Number of lod level of Input(X) should not be "
                        "greater than 1.");

      PADDLE_ENFORCE(x_lod.size() == y_lod.size() || x_lod.size() == 0,
                     "Number of lod level of Input(X) either equal to 0 "
                     "or equal to that of Input(Y).");

      int64_t out_first_dim = 0;
      if (y_lod[ref_level].size() < 1) {
        out_first_dim = x_dims[0];
      } else {
        if (x_lod.size() == 1) {  // X is LoDTensor
          for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
            int x_seq_len = x_lod[0][i] - x_lod[0][i - 1];
            out_first_dim +=
                (y_lod[ref_level][i] - y_lod[ref_level][i - 1]) * x_seq_len;
          }
        } else {  // X is normal Tensor
          for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
            out_first_dim += y_lod[ref_level][i] - y_lod[ref_level][i - 1];
          }
        }
      }
      ctx->SetOutputDim("Out", {out_first_dim, x_dims[1]});
    } else {
      framework::VarDesc* in_reader =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("Y")[0]);
      int lod_level_num = in_reader->GetLoDLevels().size();

      PADDLE_ENFORCE_GE(ref_level, 0,
                        "Level of referred lod should be greater or "
                        "equal to 0.");

      PADDLE_ENFORCE_LT(ref_level, lod_level_num,
                        "Level of referred lod should be smaller than "
                        "level number of Input(Y).");

      ctx->SetOutputDim("Out", {-1, x_dims[1]});
    }
  }
};

class SequenceExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceExpandOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) A 2-D LoDTensor whose lod "
             "level is at most 1.");
    AddInput("Y",
             "(LoDTensor, default LoDTensor<float>) Referred LoDTensor whose "
             "lod (specified level) is referred by Input(X).");
    AddOutput("Out",
              "(LodTensor, default LoDTensor<float>) Output LoDTensor which is "
              "generated from Input(X) by referring lod of Input(Y).");
    AddAttr<int>("ref_level", "Specify lod level of Input(Y).");
    AddComment(R"DOC(
Sequence Expand Operator.

This operator expands input(X) according to LOD of input(Y).
Following are cases to better explain how this works:
Case 1:

Given a 2-level LoDTensor input(X)
    X.lod = [[0,       2, 3],
             [0, 1,    3, 4]]
    X.data = [a, b, c, d]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0,    2,    4],
             [0, 3, 6, 7, 8]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 2-level LoDTensor
    Out.lod = [[0,                2,    4],
               [0,       3,       6, 7, 8]]
    Out.data = [a, a, a, b, b, b, c, d]
    Out.dims = [8, 1]

Case 2:

Given a common Tensor input(X)
    X.data = [a, b, c]
    X.dims = [3, 1]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 1-level LoDTensor
    Out.lod = [[0,    2, 3,      6]]
    Out.data = [a, a, b, c, c, c]
    Out.dims = [6, 1]

Case 3:

Given a common Tensor input(X)
    X.data = [[a, b], [c, d], [e, f]]
    X.dims = [3, 2]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 1-level LoDTensor
    Out.lod = [[0,           2,     3,                     6]]
    Out.data = [[a,b], [a,b] [c,d], [e, f], [e, f], [e, f]]
    Out.dims = [6, 2]

Case 4:

Given 2-level a LoDTensor input(X)
    X.lod = [[0,       2, 3],
             [0, 1,    3, 4]]
    X.data = [a, b, c, d]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0,    2,    4],
             [0, 3, 6, 6, 8]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 2-level LoDTensor
    Out.lod = [[0,                2,    4],
               [0,       3,       6, 6, 8]]
    Out.data = [a, a, a, b, b, b, d, d]
    Out.dims = [8, 1]


)DOC");
  }
};

class SequenceExpandOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Out"), "Input(Out) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_expand, ops::SequenceExpandOp, ops::SequenceExpandOpMaker,
            sequence_expand_grad, ops::SequenceExpandOpGrad);
REGISTER_OP_CPU_KERNEL(
    sequence_expand,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    sequence_expand_grad,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
