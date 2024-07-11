// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/rms_norm_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void RmsNormGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const paddle::optional<DenseTensor>& bias,
                       const paddle::optional<DenseTensor>& residual,
                       const DenseTensor& norm_weight,
                       const paddle::optional<DenseTensor>& norm_bias,
                       const DenseTensor& inv_var,
                       const DenseTensor& dy,
                       const float epsilon,
                       const int begin_norm_axis,
                       const float quant_scale,
                       DenseTensor* grad_x,
                       DenseTensor* grad_norm_weight,
                       DenseTensor* grad_norm_bias) {
  if (bias || residual) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "bias or residual is not supported in XPU rms_norm_grad yet"));
  }
  if (quant_scale > 0.0f) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "quantization is not supported in XPU rms_norm_grad yet"));
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  const T* x_data = x.data<T>();
  const T* norm_weight_data = norm_weight.data<T>();
  const float* invvar = inv_var.data<float>();
  const T* dy_data = dy.data<T>();
  dev_ctx.template Alloc<T>(grad_x);
  T* grad_x_data = grad_x->data<T>();
  dev_ctx.template Alloc<T>(grad_norm_weight);
  T* grad_norm_weight_data = grad_norm_weight->data<T>();
  T* grad_norm_bias_data = nullptr;
  if (norm_bias) {
    dev_ctx.template Alloc<T>(grad_norm_bias);
    grad_norm_bias_data = grad_norm_bias->data<T>();
  }

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  PADDLE_ENFORCE_EQ(
      cols,
      norm_weight.dims()[0],
      phi::errors::InvalidArgument(
          "The product from begin_norm_axis to last_axis of input tensor x, "
          "i.e., cols(%d)"
          "must be equal to the norm_weight tensor's dimension(%d).",
          cols,
          norm_weight.dims()[0]));

  int r = baidu::xpu::api::rms_layer_norm_grad<XPUType, XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x_data),
      reinterpret_cast<const XPUType*>(dy_data),
      reinterpret_cast<XPUType*>(grad_x_data),
      rows,
      cols,
      epsilon,
      reinterpret_cast<const XPUType*>(norm_weight_data),
      invvar,
      reinterpret_cast<XPUType*>(grad_norm_weight_data),
      reinterpret_cast<XPUType*>(grad_norm_bias_data),
      true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "rms_layer_norm_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(rms_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::RmsNormGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
