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

#include <memory>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/amp/check_finite_and_unscale_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CheckFiniteAndUnscaleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    found_inf->mutable_data<bool>(ctx.GetPlace());

    bool found_inf_data = false;

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // step1: inverse scale(RealDiv)
    Tensor const_tensor;
    const_tensor.mutable_data<T>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<T>(&const_tensor, static_cast<T>(1.0));

    // Inverse(1.0/scale)
    Tensor* tmp_inverse_out = const_cast<Tensor*>(scale);
    Tensor inverse_out(scale->type());
    inverse_out.Resize(scale->dims());
    inverse_out.mutable_data<T>(ctx.GetPlace());
    auto runner_inverse =
        NpuOpRunner("Div", {const_tensor, *scale}, {inverse_out}, {});
    runner_inverse.Run(stream);
    tmp_inverse_out = &inverse_out;

    // NOTE(zhiqiu):
    Tensor float_status, addr;
    float_status.mutable_data<float>({8}, ctx.GetPlace());
    addr.mutable_data<float>({8}, ctx.GetPlace());
    platform::NPUMemsetAsync(static_cast<void*>(float_status.data<T>()), 0,
                             float_status.numel() * sizeof(T), stream);
    platform::NPUMemsetAsync(static_cast<void*>(addr.data<T>()), 0,
                             addr.numel() * sizeof(T), stream);
    auto runner_float_status =
        NpuOpRunner("NPUGetFloatStatus", {addr}, {float_status},
                    {{"message", std::string("check_nan_and_inf")}});
    runner_float_status.Run(stream);

    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      out->mutable_data<T>(ctx.GetPlace());
      if (!found_inf_data) {
        // MatMul
        auto runner_matmul =
            NpuOpRunner("Mul", {*x, *tmp_inverse_out}, {*out}, {});
        runner_matmul.Run(stream);
      } else {
        // ZerosLike
        auto runner_zeroslike = NpuOpRunner("ZerosLike", {*x}, {*out}, {});
        runner_zeroslike.Run(stream);
      }
    }

    // set found_inf to true
    if (found_inf_data) {
      Tensor found_inf_tensor;
      found_inf_tensor.Resize({1});
      bool* is_found_inf =
          found_inf_tensor.mutable_data<bool>(paddle::platform::CPUPlace());
      *is_found_inf = true;

      framework::TensorCopy(
          found_inf_tensor, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), found_inf);
      ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
    }

    auto runner_clear_status =
        NpuOpRunner("NPUClearFloatStatus", {addr}, {float_status});
    runner_float_status.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(check_finite_and_unscale,
                       ops::CheckFiniteAndUnscaleNPUKernel<float>,
                       ops::CheckFiniteAndUnscaleNPUKernel<plat::float16>);
