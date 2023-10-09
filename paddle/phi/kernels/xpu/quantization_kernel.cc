// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
template <typename TX, typename TY, typename Context>
void QuantizeKernelImpl(const Context& ctx,
                        const DenseTensor& x,
                        const paddle::optional<DenseTensor>& max,
                        DenseTensor* y) {
  using XPUInX = typename XPUTypeTrait<TX>::Type;
  using XPUOutY = typename XPUTypeTrait<TY>::Type;

  auto* y_data = ctx.template Alloc<TY>(y);
  const auto* x_data = x.data<TX>();
  int64_t len = x.numel();
  const float* max_data =
      max.get_ptr() == nullptr ? nullptr : max->data<float>();
  int r = xpu::quantization<XPUInX, XPUOutY>(
      ctx.x_context(),
      reinterpret_cast<const XPUInX*>(x_data),
      reinterpret_cast<XPUOutY*>(y_data),
      len,
      max_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "quantization");
}

template <typename T, typename Context>
void QuantizeKernel(const Context& ctx,
                    const DenseTensor& x,
                    const paddle::optional<DenseTensor>& max,
                    DataType out_dtype,
                    float scale,
                    DenseTensor* y) {
  switch (out_dtype) {
    case DataType::INT16:
      QuantizeKernelImpl<T, int16_t, Context>(ctx, x, max, y);
      break;
    case DataType::INT8:
      QuantizeKernelImpl<T, int8_t, Context>(ctx, x, max, y);
      break;
    default:
      PADDLE_THROW(phi::errors::Unavailable(
          "Not supported quantize data type from %d -> %d ",
          x.dtype(),
          out_dtype));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(quantize_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::QuantizeKernel,
                   float,
                   phi::dtype::float16) {}
