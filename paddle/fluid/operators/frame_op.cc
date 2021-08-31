// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/frame_op.h"

namespace paddle {
namespace operators {

class FrameOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "frame");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "frame");

    const auto in_dims = ctx->GetInputDim("X");
    const int frame_length = ctx->Attrs().Get<int>("frame_length");
    const int hop_length = ctx->Attrs().Get<int>("hop_length");
    const int axis = ctx->Attrs().Get<int>("axis");

    std::vector<int64_t> output_shape;
    int seq_length;
    int n_frames;

    int start_axis;
    int end_axis;

    if (axis == 0) {
      seq_length = in_dims[0];
      start_axis = 1;
      end_axis = in_dims.size() - 1;
    } else {
      seq_length = in_dims[in_dims.size() - 1];
      start_axis = 0;
      end_axis = in_dims.size() - 2;
    }

    // It won't go into for loop when in_dims.size() == 1U.
    for (int i = start_axis; i <= end_axis; i++) {
      output_shape.push_back(in_dims[i]);
    }

    n_frames = 1 + (seq_length - frame_length) / hop_length;

    if (axis == 0) {
      // (n_frames, frame_length, ...)
      output_shape.insert(output_shape.begin(), frame_length);
      output_shape.insert(output_shape.begin(), n_frames);
    } else {
      // (..., frame_length, n_frames)
      output_shape.push_back(frame_length);
      output_shape.push_back(n_frames);
    }

    // VLOG(0) << "[FrameOp][InferShape]: "
    //         << framework::make_ddim(output_shape).to_str();
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

class FrameOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of abs op.");
    AddOutput("Out", "(Tensor), The output tensor of abs op.");
    AddAttr<int>("frame_length",
                 "Frame Length"
                 "Other doc of frame length arg...")
        .SetDefault(1);
    AddAttr<int>("hop_length",
                 "Hop Length"
                 "Other doc of hop length arg...")
        .SetDefault(1);
    AddAttr<int>("axis",
                 "Axis"
                 "Other doc of axis arg...")
        .SetDefault(0);
    AddComment(R"DOC(
    Frame Operator.

    This operator is used to slice frame of input $X$.

    )DOC");
  }
};

class FrameOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "frame_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "frame_grad");
    auto x_dims = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      // VLOG(0) << "[FrameOpGrad][InferShape]: " << x_dims.to_str();
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FrameOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("frame_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(frame, ops::FrameOp, ops::FrameOpMaker,
                  ops::FrameOpGradMaker<paddle::framework::OpDesc>,
                  ops::FrameOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(frame_grad, ops::FrameOpGrad);

REGISTER_OP_CPU_KERNEL(
    frame, ops::FrameKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<float>>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    frame_grad, ops::FrameGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<float>>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<double>>);
