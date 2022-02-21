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

#pragma once

#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/real_grad_kernel.h"

namespace phi {

template <typename T, typename DeviceContext>
void RealGradKernel(const DeviceContext& dev_ctx,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  auto numel = dout.numel();
  auto* dout_data = dout.data<phi::funcs::Real<T>>();
  auto* dx_data =
      dev_ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));

  paddle::platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
  phi::funcs::RealToComplexFunctor<T> functor(dout_data, dx_data, numel);
  for_range(functor);
}

}  // namespace phi
