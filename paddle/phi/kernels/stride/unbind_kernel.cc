// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/unbind_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/slice_kernel.h"

namespace phi {

template <typename Context>
void UnbindStridedKernel(const Context& ctx,
                         const DenseTensor& x,
                         int axis,
                         std::vector<DenseTensor*> outs) {
  int64_t input_dim_size = x.dims().size();
  int64_t num = outs.size();
  int64_t start = 0;

  for (int64_t i = 0; i < num; i++) {
    auto size = outs[i]->dims()[axis];
    SliceStridedKernel<Context>(
        dev_ctx, x, {start}, {start + size}, {}, {}, outs[i]);
    start += size;
  }
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    unbind, STRIDED, phi::UnbindStridedKernel) {}
