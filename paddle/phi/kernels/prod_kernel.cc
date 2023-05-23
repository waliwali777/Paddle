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

#include "paddle/phi/kernels/prod_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ProdInferKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  ProdKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(prod_infer,
                   CPU,
                   ALL_LAYOUT,
                   phi::ProdInferKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(prod_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::ProdInferKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

#if defined(PADDLE_WITH_XPU_KP) && !defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(prod_infer, KPS, ALL_LAYOUT, phi::ProdInferKernel, float) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(prod_infer, XPU, ALL_LAYOUT, phi::ProdInferKernel, float) {}
#endif
