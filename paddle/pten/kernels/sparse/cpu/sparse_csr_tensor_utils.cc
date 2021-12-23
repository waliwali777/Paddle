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

#include "paddle/pten/kernels/sparse/cpu/sparse_csr_tensor_utils.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/hybird/sparse/cpu/sparse_utils.h"

namespace pten {

template <typename T>
void ToSparseCsr(const CPUContext& dev_ctx,
                 const DenseTensor& src,
                 SparseCsrTensor* dst) {
  PADDLE_ENFORCE_EQ(src.dims().size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D Tensor."));

  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();

  int64_t non_zero_num = get_non_zero_num<T>(src, 2);
  dst->Resize(src_dims, non_zero_num);

  int64_t* crows_data = dst->mutable_non_zero_crows();
  int64_t* cols_data = dst->mutable_non_zero_cols();
  T* values_data = dst->mutable_non_zero_elements<T>();

  int non_zero_count = 0;
  for (int i = 0; i < src_dims[0]; i++) {
    crows_data[i] = non_zero_count;
    for (int j = 0; j < src_dims[1]; j++) {
      const T data = src_data[i * src_dims[1] + j];
      if (data != static_cast<T>(0)) {
        cols_data[non_zero_count] = j;
        values_data[non_zero_count] = data;
        ++non_zero_count;
      }
    }
  }
  crows_data[src_dims[0]] = non_zero_count;
}

template <typename T>
void SparseCooToCsr(const CPUContext& dev_ctx,
                    const SparseCooTensor& src,
                    SparseCsrTensor* dst) {
  const auto& dense_dims = src.dims();
  PADDLE_ENFORCE_EQ(dense_dims.size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D matrix"));
  const int64_t non_zero_num = src.nnz();
  const auto cpu_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace());
  DenseTensorMeta crows_meta(DataType::INT64,
                             paddle::framework::make_ddim({dense_dims[0] + 1}),
                             DataLayout::NCHW);

  dst->Resize(src.dims(), non_zero_num);
  int64_t* csr_crows_data = dst->mutable_non_zero_crows();
  int64_t* csr_cols_data = dst->mutable_non_zero_cols();
  T* csr_values_data = dst->mutable_non_zero_elements<T>();

  const auto& src_indices = src.non_zero_indices();
  const auto& src_values = src.non_zero_elements();
  const int64_t* src_rows_data = src_indices.data<int64_t>();
  const int64_t* src_cols_data = src_rows_data + non_zero_num;
  const T* src_values_data = src_values.data<T>();

  // 1. call src.coalesced() for distinct and sort the indices
  // 2. transform the src to dst
  if (non_zero_num <= 0) return;
  int offset = 1;
  csr_crows_data[0] = 0;
  csr_cols_data[0] = src_cols_data[0];
  csr_values_data[0] = src_values_data[0];
  for (int64_t i = 1, j = 1; i < non_zero_num; i++) {
    csr_cols_data[i] = src_cols_data[i];
    csr_values_data[i] = src_values_data[i];
    if (src_rows_data[i] == src_rows_data[i - 1]) {
    } else {
      csr_crows_data[j++] = offset;
    }
    offset += 1;
  }
  csr_crows_data[dense_dims[0]] = non_zero_num;
}

template <typename T>
void SparseCsrToDense(const CPUContext& dev_ctx,
                      const SparseCsrTensor& src,
                      DenseTensor* dst) {
  const DenseTensor& non_zero_crows = src.non_zero_crows();
  const DenseTensor& non_zero_cols = src.non_zero_cols();
  const DenseTensor& non_zero_elements = src.non_zero_elements();

  const auto& dense_dims = src.dims();

  T* out_data = dst->mutable_data<T>();
  memset(out_data, 0, sizeof(T) * dst->numel());

  const auto rows = non_zero_crows.numel() - 1;
  const auto cols = dense_dims[1];
  const int64_t* crows_data = non_zero_crows.data<int64_t>();
  const int64_t* cols_data = non_zero_cols.data<int64_t>();
  const T* elements_data = non_zero_elements.data<T>();
  int non_zero_num = 0;
  for (auto row = 0; row < rows; row++) {
    for (int64_t i = crows_data[row]; i < crows_data[row + 1]; i++) {
      out_data[row * cols + cols_data[non_zero_num]] =
          elements_data[non_zero_num];
      ++non_zero_num;
    }
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(
    to_sparse_csr, CPU, ALL_LAYOUT, pten::ToSparseCsr, float, double) {}
PT_REGISTER_KERNEL(
    sparse_coo_to_csr, CPU, ALL_LAYOUT, pten::SparseCooToCsr, float, double) {}
PT_REGISTER_KERNEL(sparse_csr_to_dense,
                   CPU,
                   ALL_LAYOUT,
                   pten::SparseCsrToDense,
                   float,
                   double) {}
