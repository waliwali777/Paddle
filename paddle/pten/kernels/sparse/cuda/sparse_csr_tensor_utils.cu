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

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/sparse.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/cuda/utils.h"
#include "paddle/pten/kernels/sparse/cuda/sparse_csr_tensor_utils.h"

namespace pten {

template <typename T>
void ToSparseCsr(const CUDAContext& dev_ctx,
                 const DenseTensor& src,
                 SparseCsrTensor* dst) {
  PADDLE_ENFORCE_EQ(src.dims().size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D Tensor."));

  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();

  const auto cpu_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace());
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(src.place());
  auto nnz_dims = paddle::framework::make_ddim({src_dims[0] + 1});
  DenseTensorMeta nnz_meta(DataType::INT32, nnz_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> nnz_ptr(new DenseTensor(allocator, nnz_meta));
  std::unique_ptr<DenseTensor> cpu_nnz_ptr(
      new DenseTensor(cpu_alloc, nnz_meta));

  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(dev_ctx);
  int* nnz = nnz_ptr->mutable_data<int32_t>();
  const int M = static_cast<int>(src_dims[0]);
  const int N = static_cast<int>(src_dims[1]);
  sparse.nnz(M, N, src_data, nnz, nnz + 1);
  pten::Copy(dev_ctx, *nnz_ptr, true, cpu_nnz_ptr.get());
  const int64_t non_zero_num = cpu_nnz_ptr->data<int>()[0];

  auto non_zero_dims = paddle::framework::make_ddim({non_zero_num});
  auto crows_dims = paddle::framework::make_ddim({src_dims[0] + 1});
  DenseTensorMeta crows_meta(DataType::INT64, crows_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> crows_ptr(
      new DenseTensor(allocator, crows_meta));
  DenseTensorMeta cols_meta(DataType::INT64, non_zero_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> cols_ptr(new DenseTensor(allocator, cols_meta));
  DenseTensorMeta values_meta(src.dtype(), non_zero_dims, src.layout());
  std::unique_ptr<DenseTensor> values_ptr(
      new DenseTensor(allocator, values_meta));

  int64_t* crows_data = crows_ptr->mutable_data<int64_t>();
  int64_t* cols_data = cols_ptr->mutable_data<int64_t>();
  T* values_data = values_ptr->mutable_data<T>();

  sparse.DenseToSparseCsr(static_cast<int>(src_dims[0]),
                          static_cast<int>(src_dims[1]),
                          src_data,
                          crows_data,
                          cols_data,
                          values_data);

  dst->SetMemberTensor(std::move(crows_ptr),
                       std::move(cols_ptr),
                       std::move(values_ptr),
                       src_dims);
}

template <typename T>
void SparseCsrToDense(const CUDAContext& dev_ctx,
                      const SparseCsrTensor& src,
                      DenseTensor* dst) {
  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(dev_ctx);
  const auto src_dims = src.dims();
  const int M = src_dims[0];
  const int N = src_dims[1];
  const DenseTensor& crows = src.non_zero_crows();
  const DenseTensor& cols = src.non_zero_cols();
  const DenseTensor& values = src.non_zero_elements();
  const int64_t nnz = src.nnz();
  sparse.SparseCsrToDense(M,
                          N,
                          nnz,
                          crows.data<int64_t>(),
                          cols.data<int64_t>(),
                          values.data<T>(),
                          dst->mutable_data<T>());
}

}  // namespace pten

PT_REGISTER_KERNEL(to_sparse_csr, CUDA, ANY, pten::ToSparseCsr, float, double) {
}
PT_REGISTER_KERNEL(
    sparse_csr_to_dense, CUDA, ANY, pten::SparseCsrToDense, float, double) {}
