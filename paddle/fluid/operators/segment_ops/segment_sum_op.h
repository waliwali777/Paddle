/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/segment_ops/segment_pooling.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T, typename IndexT>
void SegmentKernelLaunchHelper(const framework::ExecutionContext& context) {
  auto* input = context.Input<Tensor>("X");
  auto* segment = context.Input<Tensor>("SegmentIds");
  auto* output = context.Output<Tensor>("Out");
  std::string pooltype = context.Attr<std::string>("pooltype");

  int64_t num_indices = segment->numel();
  PADDLE_ENFORCE_EQ(
      num_indices, input->dims()[0],
      platform::errors::InvalidArgument(
          "segment_ids should be the same size as dimension 0 of input X."));

  if (input->numel() == 0 || segment->numel() == 0) {
    return;
  }

  bool cpu_place = context.GetPlace() == platform::CPUPlace();
  if (cpu_place) {
    auto dims = input->dims();
    auto* segment_ids = segment->data<IndexT>();
    dims[0] = static_cast<int64_t>(segment_ids[segment->numel() - 1] + 1);
    PADDLE_ENFORCE_GT(
        dims[0], 0,
        platform::errors::InvalidArgument(
            "segment ids must be >= 0, but got last id %d", dims[0]));
    output->Resize({dims});
    output->mutable_data<T>(context.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, output, static_cast<T>(0));
  }
#ifdef PADDLE_WITH_CUDA
  if (!cpu_place) {
    Tensor length;
    length.mutable_data<IndexT>(framework::make_ddim({1}),
                                platform::CPUPlace());
    IndexT* length_data = length.data<IndexT>();
    const IndexT* segment_ids = segment->data<IndexT>();

    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(length_data, segment_ids + num_indices - 1, sizeof(IndexT),
                   cudaMemcpyDeviceToHost));

    IndexT length_host = length_data[0];
    length_host++;
    PADDLE_ENFORCE_GT(
        length_host, 0,
        platform::errors::InvalidArgument(
            "segment ids must be >= 0, but got last id %d", length_data[0]));
    auto dims = input->dims();
    dims[0] = static_cast<int64_t>(length_host);
    output->Resize({dims});
    output->mutable_data<T>(context.GetPlace());
    T init_value = 0;
    if (pooltype == "MAX") {
      init_value = static_cast<T>(-FLT_MAX);
    } else if (pooltype == "MIN") {
      init_value = static_cast<T>(FLT_MAX);
    }
    math::SetConstant<DeviceContext, T> setconst;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    setconst(dev_ctx, output, static_cast<T>(init_value));
  }
#endif

  SegmentPoolFunctor<DeviceContext, T, IndexT> pool;
  Tensor* index = nullptr;
  pool(context.template device_context<DeviceContext>(), *input, *segment,
       output, index, pooltype);
}

template <typename DeviceContext, typename T>
class SegmentSumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* segment = context.Input<Tensor>("SegmentIds");
    auto index_type = segment->type();
    if (index_type == framework::proto::VarType::INT32) {
      SegmentKernelLaunchHelper<DeviceContext, T, int>(context);
    } else if (index_type == framework::proto::VarType::INT64) {
      SegmentKernelLaunchHelper<DeviceContext, T, int64_t>(context);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported index type, Expected int, int64, but got %s.",
          index_type));
    }
  }
};

template <typename DeviceContext, typename T>
class SegmentSumGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Input<Tensor>("Out");
    auto* segment = context.Input<Tensor>("SegmentIds");
    auto* out_g = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<Tensor>(framework::GradVarName("X"));
    std::string pooltype = context.Attr<std::string>("pooltype");

    const Tensor* index = nullptr;
    in_g->mutable_data<T>(context.GetPlace());

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, in_g, static_cast<T>(0));

    auto index_type = segment->type();
    if (index_type == framework::proto::VarType::INT32) {
      SegmentPoolGradFunctor<DeviceContext, T, int> pool;
      pool(context.template device_context<DeviceContext>(), *input, *output,
           *out_g, *segment, in_g, index, pooltype);
    } else if (index_type == framework::proto::VarType::INT64) {
      SegmentPoolGradFunctor<DeviceContext, T, int64_t> pool;
      pool(context.template device_context<DeviceContext>(), *input, *output,
           *out_g, *segment, in_g, index, pooltype);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported index type, Expected int, int64, but got %s.",
          index_type));
    }
  }
};

}  // namespace operators
}  // namespace paddle
