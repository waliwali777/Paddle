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

#include <algorithm>
#include <vector>
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/abs_kernel.h"
#include "paddle/pten/kernels/funcs/complex_functors.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

namespace pten {

template <typename T, typename Enable = void>
struct CudaAbsFunctor;

template <typename T>
struct CudaAbsFunctor<T, pten::funcs::Complex<T, pten::funcs::Real<T>>> {
  __device__ __forceinline__ pten::funcs::Real<T> operator()(const T x) const {
    return abs(x);
  }
};

template <typename T>
struct CudaAbsFunctor<T, pten::funcs::NoComplex<T, pten::funcs::Real<T>>> {
  __device__ __forceinline__ T operator()(const T x) const {
    return std::abs(x);
  }
};

template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<pten::funcs::Real<T>>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  auto functor = CudaAbsFunctor<T>();

  funcs::ElementwiseKernel<pten::funcs::Real<T>>(ctx, ins, &outs, functor);
}

}  // namespace pten

PT_REGISTER_KERNEL(abs,
                   GPU,
                   ALL_LAYOUT,
                   pten::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   pten::dtype::float16,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}
