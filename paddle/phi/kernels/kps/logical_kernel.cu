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
// limitation

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/kernels/logical_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/logical_functor.h"

namespace phi {

#define DEFINE_LOGICAL_BINARY_KERNEL(type)                               \
  template <typename T, typename Context>                                \
  void Logical##type##Kernel(const Context& dev_ctx,                     \
                             const DenseTensor& x,                       \
                             const DenseTensor& y,                       \
                             DenseTensor* out) {                         \
    using InT = typename funcs::Logical##type##Functor<T>::ELEMENT_TYPE; \
    using OutT = bool;                                                   \
    dev_ctx.template Alloc<bool>(out);                                   \
    funcs::Logical##type##Functor<T> binary_func;                        \
    std::vector<const DenseTensor*> ins = {&x, &y};                      \
    std::vector<DenseTensor*> outs = {out};                              \
    funcs::BroadcastKernel<ElementwiseType::kBinary, InT, OutT>(         \
        dev_ctx, ins, &outs, -1, binary_func);                           \
  }

DEFINE_LOGICAL_BINARY_KERNEL(And)
DEFINE_LOGICAL_BINARY_KERNEL(Or)
DEFINE_LOGICAL_BINARY_KERNEL(Xor)
#undef DEFINE_LOGICAL_BINARY_KERNEL

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  using InT = typename funcs::LogicalNotFunctor<T>::ELEMENT_TYPE;
  using OutT = bool;

  dev_ctx.template Alloc<bool>(out);
  funcs::LogicalNotFunctor<T> unary_func;
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  funcs::BroadcastKernel<ElementwiseType::kUnary, InT, OutT>(
      dev_ctx, ins, &outs, -1, unary_func);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    logical_and, KPS, ALL_LAYOUT, phi::LogicalAndKernel, int /*, float*/) {}
PD_REGISTER_KERNEL(
    logical_Or, KPS, ALL_LAYOUT, phi::LogicalOrKernel, int /*, float*/) {}
PD_REGISTER_KERNEL(
    logical_Not, KPS, ALL_LAYOUT, phi::LogicalNotKernel, int /*, float*/) {}
PD_REGISTER_KERNEL(
    logical_Xor, KPS, ALL_LAYOUT, phi::LogicalXorKernel, int /*, float*/) {}
#endif
