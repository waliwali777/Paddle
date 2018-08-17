//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sequence_enumerate_op.h"

namespace paddle {
namespace operators {

class SequenceEnumerateOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of SequecceEnumerate operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(X) of SequenceEnumerate operator should not be null.");

    const auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(
        x_dims.size() == 2 && x_dims[1] == 1,
        "Input(X) of SequenceEnumerate operator should be a 2-D LoDTensor "
        "with the 2nd dimension equal to 1.");

    const auto win_size = ctx->Attrs().Get<int>("win_size");
    PADDLE_ENFORCE(win_size <= x_dims[0],
                   "The enumerate window size should be less than or equal to "
                   "input sequence length.");
    ctx->SetOutputDim("Out", {x_dims[0], win_size});
    ctx->ShareLoD("X", "Out");
  }
};

class SequenceEnumerateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(2-D LoDTensor with the 2nd dimension equal to 1) "
             "Input LoDTensor of SequenceEnumerate operator.");
    AddOutput("Out",
              "(2-D LoDTensor with the 2nd dimension equal to 1) "
              "Output LoDTensor of SequenceEnumerate operator.");
    AddAttr<int>("win_size", "(int) The enumerate sequence window size.")
        .AddCustomChecker([](const int& win_size) {
          PADDLE_ENFORCE(win_size >= 2,
                         "The window size should be greater than 2.");
        });
    AddAttr<int>("pad_value", "(int) The enumerate sequence padding value.")
        .SetDefault(0);
    AddComment(R"DOC(
Sequence Enumerate Operator.

Sequence enumerate operator generate a new LoDTensor 
with the same 1st dimension length as the original LoDTensor, 
and with the 2nd dimension equal to the input window length, 
the new sub-sequence on 2nd dimension is enumerated one by one on the original sequence.
The values of the last insufficient part areall filled with the input pad_value.

Examples:
Case 1:
  Input:
    X.lod = [[0, 3, 5]]
    X.data = [1, 2, 3, 4, 5]
    X.dims = [5, 1]
  Attrs:
    win_size = 2
    pad_value = 0
  Output:
    Out.lod = [[0, 3, 5]]
    Out.data = [[1, 2], [2, 3], [3, 4], [4, 5], [0, 0]]
    Out.dims = [5, 2]

  Currently, only 1-level LoDTensor is supported.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sequence_enumerate, ops::SequenceEnumerateOp,
                             ops::SequenceEnumerateOpMaker);
REGISTER_OP_CPU_KERNEL(
    sequence_enumerate,
    ops::SequenceEnumerateKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::SequenceEnumerateKernel<paddle::platform::CPUDeviceContext, int64_t>);
