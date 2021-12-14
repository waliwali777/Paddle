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

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather_scatter_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class TakeAlongAxisOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto input = ctx.Input<Tensor>("Input");
    VLOG(3) << " input:" << *(input->data<T>());
    auto axis = ctx.Attr<int>("Axis");
    auto index = ctx.Input<Tensor>("Index");
    auto result = ctx.Output<Tensor>("Result");
    result->Resize(index->dims());
    result->mutable_data<T>(ctx.GetPlace());
    // // resize_output(result, index.sizes());
    // // check_no_internal_overlap(self, result);
    // // check_no_partial_overlap(result, index);
    // VLOG(3) << "000000000";

    const auto &index_type = index->type();
    if (index_type == framework::proto::VarType::INT32) {
      cpu_gather_kernel<T, int32_t>(*input, axis, *index, *result,
                                    ctx.device_context());
    } else if (index_type == framework::proto::VarType::INT64) {
      cpu_gather_kernel<T, int64_t>(*input, axis, *index, *result,
                                    ctx.device_context());
    }
    VLOG(3) << "<<<< Done Compute <<<<<";
  }
};

template <typename T>
class TakeAlongAxisGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto input_d = ctx.Output<Tensor>(framework::GradVarName("Input"));
    VLOG(3) << " grad 111111:";
    // VLOG(3) << " input grad:" << *(input_d->data<T>());
    auto index = ctx.Input<Tensor>("Index");
    auto result_d = ctx.Input<Tensor>(framework::GradVarName("Result"));
    VLOG(3) << " index grad:" << *(index->data<int>());
    VLOG(3) << " result_d grad:" << *(result_d->data<T>());
    VLOG(3) << " grad 22222222:";
    auto axis = ctx.Attr<int>("Axis");
    VLOG(3) << " grad 3333333:";
    VLOG(3) << " index->dims: " << sizeof(index->dims());
    input_d->Resize(index->dims());
    VLOG(3) << " grad 4444444:";
    input_d->mutable_data<T>(ctx.GetPlace());
    VLOG(3) << " grad 5555555:";
    // // resize_output(result, index.sizes());
    // // check_no_internal_overlap(self, result);
    // // check_no_partial_overlap(result, index);
    // VLOG(3) << "000000000";

    const auto &index_type = index->type();
    VLOG(3) << " grad 666666:";
    if (index_type == framework::proto::VarType::INT32) {
      VLOG(3) << " grad 77777:";
      cpu_scatter_assign_kernel<T, int32_t>(
          *input_d, axis, *index, *result_d,
          ctx.device_context());  // the gradient of gather is scatter
    } else if (index_type == framework::proto::VarType::INT64) {
      VLOG(3) << " grad 88888:";
      cpu_scatter_assign_kernel<T, int64_t>(*input_d, axis, *index, *result_d,
                                            ctx.device_context());
    }
    VLOG(3) << "<<<< Done Compute <<<<<";
  }
};

}  // namespace operators
}  // namespace paddle
