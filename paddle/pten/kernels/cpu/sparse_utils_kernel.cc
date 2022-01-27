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

#include "paddle/pten/kernels/sparse_utils_kernel.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

template <typename T>
inline bool is_zero(const T* data, const size_t n) {
  const T zero = static_cast<T>(0);
  for (size_t i = 0; i < n; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline int64_t get_non_zero_num(const DenseTensor& dense,
                                const int64_t sparse_dim) {
  const auto& dims = dense.dims();
  PADDLE_ENFORCE_GE(
      dims.size(),
      sparse_dim,
      paddle::platform::errors::InvalidArgument(
          "sparse_dim(%d) should be less than or equal to dense.dim(%d)",
          sparse_dim,
          dims.size()));

  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  const T* data = dense.data<T>();
  int64_t non_zero_num = 0;
  for (int64_t i = 0; i < rows; i++) {
    if (!is_zero(data + i * cols, cols)) {
      non_zero_num = non_zero_num + 1;
    }
  }
  return non_zero_num;
}

template <typename T, typename Context>
void DenseToSparseCooKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const int64_t sparse_dim,
                            SparseCooTensor* out) {
  const T* x_data = x.data<T>();
  const auto& x_dims = x.dims();

  int64_t non_zero_num = get_non_zero_num<T>(x, sparse_dim);

  const auto place = dev_ctx.GetPlace();
  const auto values_dims = InferDenseDims(x_dims, sparse_dim, non_zero_num);
  DenseTensorMeta indices_meta(DataType::INT64,
                               {sparse_dim, static_cast<int64_t>(non_zero_num)},
                               DataLayout::NCHW);
  DenseTensorMeta values_meta(x.meta().dtype, values_dims, x.meta().layout);
  pten::DenseTensor indices =
      pten::Empty<int64_t, Context>(dev_ctx, std::move(indices_meta));
  pten::DenseTensor values =
      pten::Empty<T, Context>(dev_ctx, std::move(values_meta));
  int64_t* indices_data = indices.mutable_data<int64_t>(place);
  T* values_data = values.mutable_data<T>(place);

  auto dims_2d = flatten_to_2d(x_dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  int index = 0;
  for (int i = 0; i < rows; i++) {
    if (!is_zero(x_data + i * cols, cols)) {
      int64_t sparse_index = i;
      for (int64_t j = sparse_dim - 1; j >= 0; j--) {
        indices_data[j * non_zero_num + index] = sparse_index % x_dims[j];
        sparse_index /= x_dims[j];
      }
      memcpy(values_data + index * cols, x_data + i * cols, cols * sizeof(T));
      ++index;
    }
  }
  out->SetMember(indices, values, x_dims, true);
}

}  // namespace pten

PT_REGISTER_KERNEL(dense_to_sparse_coo,
                   CPU,
                   ALL_LAYOUT,
                   pten::DenseToSparseCooKernel,
                   float,
                   double) {}
