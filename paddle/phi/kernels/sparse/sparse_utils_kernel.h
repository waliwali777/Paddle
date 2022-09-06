/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void DenseToCooKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const int64_t sparse_dim,
                      SparseCooTensor* out);

template <typename T, typename Context>
SparseCooTensor DenseToCoo(const Context& dev_ctx,
                           const DenseTensor& x,
                           const int64_t sparse_dim) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  DenseToCooKernel<T, Context>(dev_ctx, x, sparse_dim, &coo);
  return coo;
}

template <typename T, typename Context>
void CsrToCooKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    SparseCooTensor* out);

template <typename T, typename Context>
SparseCooTensor CsrToCoo(const Context& dev_ctx, const SparseCsrTensor& x) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  CsrToCooKernel<T, Context>(dev_ctx, x, &coo);
  return coo;
}

template <typename T, typename Context>
void CooToCsrKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    SparseCsrTensor* out);

template <typename T, typename Context>
SparseCsrTensor CooToCsr(const Context& dev_ctx, const SparseCooTensor& x) {
  DenseTensor crows;
  DenseTensor cols;
  DenseTensor non_zero_elements;
  SparseCsrTensor csr(crows, cols, non_zero_elements, x.dims());
  CooToCsrKernel<T, Context>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename Context>
void DenseToCsrKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      SparseCsrTensor* out) {
  const auto& x_dims = x.dims();
  bool valid = x_dims.size() == 2 || x_dims.size() == 3;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    phi::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D Tensor."));

  const int64_t sparse_dim = x_dims.size() == 2 ? 2 : 3;
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  MetaTensor meta_out_coo(&coo);
  phi::sparse::UnchangedInferMeta(x, &meta_out_coo);
  MetaTensor meta_out(out);
  phi::sparse::UnchangedInferMeta(x, &meta_out);
  DenseToCooKernel<T, Context>(dev_ctx, x, sparse_dim, &coo);
  CooToCsrKernel<T, Context>(dev_ctx, coo, out);
}

template <typename T, typename Context>
SparseCsrTensor DenseToCsr(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor crows;
  DenseTensor cols;
  DenseTensor non_zero_elements;
  SparseCsrTensor csr(crows, cols, non_zero_elements, x.dims());
  DenseToCsrKernel<T, Context>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename Context>
void CooToDenseKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      DenseTensor* out);

template <typename T, typename Context>
DenseTensor CooToDense(const Context& dev_ctx, const SparseCooTensor& x) {
  DenseTensorMeta meta(x.dtype(), x.dims(), x.non_zero_elements().layout());
  DenseTensor dense = phi::Empty(dev_ctx, std::move(meta));
  CooToDenseKernel<T, Context>(dev_ctx, x, &dense);
  return dense;
}

template <typename T, typename Context>
void CsrToDenseKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      DenseTensor* out) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  MetaTensor meta_out_coo(&coo);
  phi::sparse::UnchangedInferMeta(x, &meta_out_coo);
  CsrToCooKernel<T, Context>(dev_ctx, x, &coo);
  MetaTensor meta_out(out);
  phi::sparse::UnchangedInferMeta(x, &meta_out);
  CooToDenseKernel<T, Context>(dev_ctx, coo, out);
}

template <typename T, typename Context>
DenseTensor CsrToDense(const Context& dev_ctx, const SparseCsrTensor& x) {
  DenseTensorMeta meta(x.dtype(), x.dims(), x.non_zero_elements().layout());
  DenseTensor dense = phi::Empty(dev_ctx, std::move(meta));
  CsrToDenseKernel<T, Context>(dev_ctx, x, &dense);
  return dense;
}

template <typename T, typename Context>
void ValuesCooKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     DenseTensor* out) {
  *out = x.non_zero_elements();
}

template <typename T, typename Context>
void ValuesCsrKernel(const Context& dev_ctx,
                     const SparseCsrTensor& x,
                     DenseTensor* out) {
  *out = x.non_zero_elements();
}

template <typename T, typename Context>
void SparseCooTensorKernel(const Context& dev_ctx,
                           const DenseTensor& values,
                           const DenseTensor& indices,
                           const IntArray& dense_shape,
                           SparseCooTensor* out) {
  *out =
      SparseCooTensor(indices, values, phi::make_ddim(dense_shape.GetData()));
  std::cout << out->dims() << std::endl;
}

}  // namespace sparse
}  // namespace phi
