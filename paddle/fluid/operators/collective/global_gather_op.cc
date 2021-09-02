/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/global_gather_op.h"

namespace paddle {
namespace operators {

class GlobalGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("x"), "Input", "x", "GlobalGather");
    OP_INOUT_CHECK(ctx->HasInput("local_count"), "Input", "local_count",
                   "GlobalGather");
    OP_INOUT_CHECK(ctx->HasInput("global_count"), "Input", "global_count",
                   "GlobalGather");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "GlobalGather");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
    // int in_feat = ctx->Attrs().Get<int>("in_feat");
    framework::DDim out_dims = framework::make_ddim({-1, -1});
    ctx->SetOutputDim("Out", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "x"), ctx.GetPlace());
  }
};

class GlobalGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("x", "(Tensor) tensor send.");
    AddInput("local_count",
             "(Tensor) Tensor which have n_expert * world_size elements that "
             "indicates"
             "how many data needed to be received.");
    AddInput("global_count",
             "(Tensor) Tensor which have n_expert * world_size elements that "
             "indicates"
             "how many data needed to be sent.");
    AddOutput("Out", "(Tensor) the result of alltoall.");
    // AddAttr<int>("in_feat", "(int default 0) The feature shape of the x.")
    //     .SetDefault(0);
    // AddAttr<int>("n_expert", "(int default 1) The number of experts in every
    // rank.")
    //     .SetDefault(1);
    // AddAttr<int>("world_size", "(int default 1) The number of rank.")
    //     .SetDefault(1);
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
Global Gather Operator
Gather data in x to n_expert * world_size exeperts according to
local_count and receive tensors from n_expert * world_size experts according
to global_count.
)DOC");
  }
};

template <typename T>
class GlobalGatherOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("global_scatter");
    retv->SetInput("x", this->OutputGrad("Out"));
    retv->SetInput("local_count", this->Input("local_count"));
    retv->SetInput("global_count", this->Input("global_count"));
    retv->SetOutput("Out", this->InputGrad("x"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(global_gather, ops::GlobalGatherOp, ops::GlobalGatherOpMaker,
                  ops::GlobalGatherOpGradMaker<paddle::framework::OpDesc>,
                  ops::GlobalGatherOpGradMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(global_gather, ops::GlobalGatherOpCPUKernel<float>,
                       ops::GlobalGatherOpCPUKernel<double>,
                       ops::GlobalGatherOpCPUKernel<int>,
                       ops::GlobalGatherOpCPUKernel<int64_t>,
                       ops::GlobalGatherOpCPUKernel<plat::float16>);
