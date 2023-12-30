//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/phi/backends/dynload/cusparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename T>
cudaDataType_t GetGpuDataType() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    return CUDA_R_16F;
  }
}

template <typename T>
cusparseIndexType_t GetCusparseIndexType() {
  if (std::is_same<T, int32_t>::value) {
    return CUSPARSE_INDEX_32I;
  } else if (std::is_same<T, int64_t>::value) {
    return CUSPARSE_INDEX_64I;
  }
}

inline cusparseOperation_t GetTransposeOperation(const bool trans) {
  if (trans) {
    return CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
}

inline cusparseSpMMAlg_t GetSpMMAlgorithm(const SparseCsrTensor& x) {
  // TODO(zhouwei): will change to 'CUSPARSE_SPMM_CSR_ALG2' when support batch
  return CUSPARSE_SPMM_CSR_ALG2;
}

inline cusparseSpMMAlg_t GetSpMMAlgorithm(const SparseCooTensor& x) {
  return CUSPARSE_SPMM_ALG_DEFAULT;
}

/************* SPARSE MATRIX DESCRIPTOR (COO/CSR) ************/

template <typename T, typename IntT>
inline void CreateCsrDescriptor(const phi::SparseCsrTensor& x,
                                const phi::GPUContext& dev_ctx,
                                cusparseSpMatDescr_t* descriptor) {
  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  auto x_ndims = xdim_vec.size();
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      phi::errors::InvalidArgument("the dim size of SparseCsrTensor must be "
                                   "greater than or eaqual to 2."));
  int64_t M = xdim_vec[x_ndims - 2];
  int64_t N = xdim_vec[x_ndims - 1];
  int batch_size = 1;
  for (int i = 0; i < x_ndims - 2; i++) {
    batch_size *= xdim_vec[i];
  }
  PADDLE_ENFORCE_EQ(x.non_zero_crows().numel(),
                    batch_size * (M + 1),
                    phi::errors::PreconditionNotMet(
                        "the length of SparseCsrTensor crows is not right."));

  const IntT* crows_data = x.non_zero_crows().data<IntT>();
  const IntT* cols_data = x.non_zero_cols().data<IntT>();
  const T* values_data = x.non_zero_elements().data<T>();
  int64_t batch_nnz = x.nnz() / batch_size;
  cudaDataType_t gpu_type = GetGpuDataType<T>();
  cusparseIndexType_t index_type = GetCusparseIndexType<IntT>();
  dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseCreateCsr(descriptor,
                                    M,
                                    N,
                                    batch_nnz,
                                    const_cast<IntT*>(crows_data),
                                    const_cast<IntT*>(cols_data),
                                    const_cast<T*>(values_data),
                                    index_type,
                                    index_type,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    gpu_type);
  });
  if (batch_size > 1) {
#if CUDA_VERSION >= 11080
    dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCsrSetStridedBatch(
          *descriptor, batch_size, M + 1, batch_nnz);
    });
#else
    PADDLE_THROW(phi::errors::Unimplemented(
        "Batch Sparse matmul use 'cusparseCsrSetStridedBatch', which is "
        "supported from CUDA 11.8"));
#endif
  }
}

template <typename T, typename IntT>
inline void CreateCooDescriptor(const phi::SparseCooTensor& x,
                                const phi::GPUContext& dev_ctx,
                                cusparseSpMatDescr_t* descriptor) {
  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  auto x_ndims = xdim_vec.size();
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      phi::errors::InvalidArgument("the dim size of SparseCsrTensor must be "
                                   "greater than or eaqual to 2."));

  int64_t M = xdim_vec[x_ndims - 2];
  int64_t N = xdim_vec[x_ndims - 1];
  int batch_size = 1;
  for (int i = 0; i < x_ndims - 2; i++) {
    batch_size *= xdim_vec[i];
  }
  int64_t nnz = x.nnz();

  const IntT* indices_data = x.non_zero_indices().data<IntT>();
  const T* values_data = x.non_zero_elements().data<T>();
  auto rows_data = indices_data + (x_ndims - 2) * nnz;
  auto cols_data = indices_data + (x_ndims - 1) * nnz;

  int64_t batch_nnz = nnz / batch_size;
  cudaDataType_t gpu_type = GetGpuDataType<T>();
  cusparseIndexType_t index_type = GetCusparseIndexType<IntT>();
  dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseCreateCoo(descriptor,
                                    M,
                                    N,
                                    batch_nnz,
                                    const_cast<IntT*>(rows_data),
                                    const_cast<IntT*>(cols_data),
                                    const_cast<T*>(values_data),
                                    index_type,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    gpu_type);
  });

  if (batch_size > 1) {
#if CUDA_VERSION >= 11080
    dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCooSetStridedBatch(
          *descriptor, batch_size, batch_nnz);
    });
#else
    PADDLE_THROW(phi::errors::Unimplemented(
        "Batch Sparse matmul use 'cusparseCooSetStridedBatch', which is "
        "supported from CUDA 11.8"));
#endif
  }
}

template <typename T>
class CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatDescriptor(const phi::SparseCsrTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        x.non_zero_crows().dtype(), "Csr CuSparseSpMatDescriptor", ([&] {
          CreateCsrDescriptor<T, data_t>(x, dev_ctx_, &descriptor_);
        }));
    VLOG(6) << "Create csr cusparseSpMatDescr_t " << &descriptor_;
  }

  explicit CuSparseSpMatDescriptor(const phi::SparseCooTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        x.non_zero_indices().dtype(), "Coo CuSparseSpMatDescriptor", ([&] {
          CreateCooDescriptor<T, data_t>(x, dev_ctx_, &descriptor_);
        }));
    VLOG(6) << "Create coo cusparseSpMatDescr_t " << &descriptor_;
  }

  ~CuSparseSpMatDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroySpMat(descriptor_);
    });
    VLOG(6) << "Destroy cusparseSpMatDescr_t " << &descriptor_;
  }

  const cusparseSpMatDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseSpMatDescr_t descriptor_;
};

/************* DENSE MATRIX DESCRIPTOR ************/
template <typename T>
class CuSparseDnMatDescriptor {
 public:
  explicit CuSparseDnMatDescriptor(const phi::DenseTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    PADDLE_ENFORCE_GE(
        x_ndims,
        2,
        phi::errors::InvalidArgument("the dim size of DenseTensor must be "
                                     "greater than or eaqual to 2."));

    int64_t M = xdim_vec[x_ndims - 2];
    int64_t N = xdim_vec[x_ndims - 1];
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }

    const T* x_data = x.data<T>();
    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateDnMat(&descriptor_,
                                        M,
                                        N,
                                        N,
                                        const_cast<T*>(x_data),
                                        gpu_type,
                                        CUSPARSE_ORDER_ROW);
    });

    PADDLE_ENFORCE_EQ(x.numel(), batch_size * M * N);
    if (batch_size > 1) {
#if CUDA_VERSION >= 11080
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseDnMatSetStridedBatch(
            descriptor_, batch_size, M * N);
      });
#else
      PADDLE_THROW(phi::errors::Unimplemented(
          "Batch Sparse matmul use 'cusparseDnMatSetStridedBatch', which is "
          "supported from CUDA 11.8"));
#endif
    }
    VLOG(6) << "Create cusparseDnMatDescr_t " << &descriptor_;
  }

  ~CuSparseDnMatDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroyDnMat(descriptor_);
    });
    VLOG(6) << "Destroy cusparseDnMatDescr_t " << &descriptor_;
  }

  const cusparseDnMatDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseDnMatDescr_t descriptor_;
};

/************* DENSE VECTOR DESCRIPTOR ************/
template <typename T>
class CuSparseDnVecDescriptor {
 public:
  explicit CuSparseDnVecDescriptor(const phi::DenseTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    PADDLE_ENFORCE_GE(x_ndims,
                      1,
                      phi::errors::InvalidArgument(
                          "the dim size of Vec must be eaqual to 1."));

    const T* x_data = x.data<T>();
    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateDnVec(
          &descriptor_, x.numel(), const_cast<T*>(x_data), gpu_type);
    });

    VLOG(6) << "Create cusparseDnVecDescr_t " << &descriptor_;
  }

  ~CuSparseDnVecDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroyDnVec(descriptor_);
    });
    VLOG(6) << "Destroy cusparseDnVecDescr_t " << &descriptor_;
  }

  const cusparseDnVecDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseDnVecDescr_t descriptor_;
};

/************* SPARSE*DENSE->DENSE MATMUL ************/
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SPMM(bool transa,
                                       bool transb,
                                       T alpha,
                                       const TensorType& mat_a,
                                       const phi::DenseTensor& mat_b,
                                       T beta,
                                       phi::DenseTensor* mat_out) const {
  auto a_descriptor = CuSparseSpMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto out_descriptor = CuSparseDnMatDescriptor<T>(*mat_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM_bufferSize(handle,
                                          GetTransposeOperation(transa),
                                          GetTransposeOperation(transb),
                                          &alpha,
                                          a_descriptor.descriptor(),
                                          b_descriptor.descriptor(),
                                          &beta,
                                          out_descriptor.descriptor(),
                                          gpu_type,
                                          GetSpMMAlgorithm(mat_a),
                                          &buffer_size);
  });

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM(handle,
                               GetTransposeOperation(transa),
                               GetTransposeOperation(transb),
                               &alpha,
                               a_descriptor.descriptor(),
                               b_descriptor.descriptor(),
                               &beta,
                               out_descriptor.descriptor(),
                               gpu_type,
                               GetSpMMAlgorithm(mat_a),
                               tmp_buffer_ptr);
  });
}

/************* SPARSE*DENSE->DENSE MV ************/
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SPMV(bool transa,
                                       T alpha,
                                       const TensorType& mat_a,
                                       const phi::DenseTensor& vec_x,
                                       T beta,
                                       phi::DenseTensor* vec_out) const {
  auto a_descriptor = CuSparseSpMatDescriptor<T>(mat_a, dev_ctx_);
  auto x_descriptor = CuSparseDnVecDescriptor<T>(vec_x, dev_ctx_);
  auto out_descriptor = CuSparseDnVecDescriptor<T>(*vec_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMV_bufferSize(handle,
                                          GetTransposeOperation(transa),
                                          &alpha,
                                          a_descriptor.descriptor(),
                                          x_descriptor.descriptor(),
                                          &beta,
                                          out_descriptor.descriptor(),
                                          gpu_type,
#if CUDA_VERSION >= 11040
                                          CUSPARSE_SPMV_ALG_DEFAULT,
#else
                                          CUSPARSE_MV_ALG_DEFAULT,
#endif
                                          &buffer_size);
  });

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMV(handle,
                               GetTransposeOperation(transa),
                               &alpha,
                               a_descriptor.descriptor(),
                               x_descriptor.descriptor(),
                               &beta,
                               out_descriptor.descriptor(),
                               gpu_type,
#if CUDA_VERSION >= 11040
                               CUSPARSE_SPMV_ALG_DEFAULT,
#else
                               CUSPARSE_MV_ALG_DEFAULT,
#endif
                               tmp_buffer_ptr);
  });
}

/************* DENSE*DENSE->SPARSE MATMUL ************/
#if CUDA_VERSION >= 11030
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SDDMM(bool transa,
                                        bool transb,
                                        T alpha,
                                        const phi::DenseTensor& mat_a,
                                        const phi::DenseTensor& mat_b,
                                        T beta,
                                        TensorType* mat_out) const {
  auto a_descriptor = CuSparseDnMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto out_descriptor = CuSparseSpMatDescriptor<T>(*mat_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_bufferSize(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_descriptor.descriptor(),
                                           b_descriptor.descriptor(),
                                           &beta,
                                           out_descriptor.descriptor(),
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           &buffer_size);
  });

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_preprocess(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_descriptor.descriptor(),
                                           b_descriptor.descriptor(),
                                           &beta,
                                           out_descriptor.descriptor(),
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           tmp_buffer_ptr);
  });

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM(handle,
                                GetTransposeOperation(transa),
                                GetTransposeOperation(transb),
                                &alpha,
                                a_descriptor.descriptor(),
                                b_descriptor.descriptor(),
                                &beta,
                                out_descriptor.descriptor(),
                                gpu_type,
                                CUSPARSE_SDDMM_ALG_DEFAULT,
                                tmp_buffer_ptr);
  });
}
#endif

/************* SPARSE*SPARSE->SPARSE MATMUL ************/
template <typename T>
class CuSparseSpGEMMCsrDescriptor {
 public:
  explicit CuSparseSpGEMMCsrDescriptor(const phi::SparseCsrTensor& x,
                                       const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();

    int64_t M = xdim_vec[x_ndims - 2];
    int64_t N = xdim_vec[x_ndims - 1];
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }

    const int32_t *crows_data, *cols_data;
    if (x.crows().dtype() == phi::DataType::INT32) {
      crows_data = x.crows().data<int32_t>();
      cols_data = x.cols().data<int32_t>();
    } else {
      phi::MetaTensor crows_meta(&crows_int);
      crows_meta.set_dims(x.crows().dims());

      phi::MetaTensor cols_meta(&cols_int);
      cols_meta.set_dims(x.cols().dims());

      phi::CastKernel<int64_t>(
          dev_ctx, x.crows(), phi::DataType::INT32, &crows_int);
      phi::CastKernel<int64_t>(
          dev_ctx, x.cols(), phi::DataType::INT32, &cols_int);

      crows_data = crows_int.data<int32_t>();
      cols_data = cols_int.data<int32_t>();
    }

    const T* values_data = x.values().data<T>();
    int64_t batch_nnz = x.nnz() / batch_size;
    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateCsr(&descriptor_,
                                      M,
                                      N,
                                      batch_nnz,
                                      const_cast<int32_t*>(crows_data),
                                      const_cast<int32_t*>(cols_data),
                                      const_cast<T*>(values_data),
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      gpu_type);
    });

    VLOG(6) << "Create csr cusparseSpMatDescr_t " << &descriptor_;
  }

  ~CuSparseSpGEMMCsrDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroySpMat(descriptor_);
    });
    VLOG(6) << "Destroy cusparseSpMatDescr_t " << &descriptor_;
  }

  const cusparseSpMatDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseSpMatDescr_t descriptor_;

  // temporarily save crows and cols for int64_t index csr
  DenseTensor crows_int;
  DenseTensor cols_int;
};

template <>
template <typename T>
void SparseBlas<phi::GPUContext>::SPGEMM(bool transa,
                                         bool transb,
                                         T alpha,
                                         const SparseCsrTensor& mat_a,
                                         const SparseCsrTensor& mat_b,
                                         T beta,
                                         SparseCsrTensor* mat_out) const {
  DenseTensor* mat_out_crows = mat_out->mutable_crows();
  DenseTensor* mat_out_cols = mat_out->mutable_cols();
  DenseTensor* mat_out_values = mat_out->mutable_values();

  MetaTensor out_crows_meta(mat_out_crows);
  out_crows_meta.set_dtype(phi::DataType::INT32);
  out_crows_meta.set_dims(mat_a.crows().dims());
  dev_ctx_.template Alloc<int32_t>(mat_out_crows);

  MetaTensor out_cols_meta(mat_out_cols);
  out_cols_meta.set_dtype(phi::DataType::INT32);
  dev_ctx_.template Alloc<int32_t>(mat_out_cols);

  MetaTensor out_values_meta(mat_out_values);
  out_values_meta.set_dtype(mat_a.values().dtype());
  dev_ctx_.template Alloc<T>(mat_out_values);

  auto a_descriptor = CuSparseSpGEMMCsrDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseSpGEMMCsrDescriptor<T>(mat_b, dev_ctx_);
  auto out_descriptor = CuSparseSpGEMMCsrDescriptor<T>(*mat_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_a_size = 0, buffer_b_size = 0;

  cusparseSpGEMMDescr_t spgemmDesc;
  phi::dynload::cusparseSpGEMM_createDescr(&spgemmDesc);
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpGEMM_workEstimation(handle,
                                                GetTransposeOperation(transa),
                                                GetTransposeOperation(transb),
                                                &alpha,
                                                a_descriptor.descriptor(),
                                                b_descriptor.descriptor(),
                                                &beta,
                                                out_descriptor.descriptor(),
                                                gpu_type,
                                                CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc,
                                                &buffer_a_size,
                                                nullptr);
  });

  phi::Allocator::AllocationPtr tmp_buffer_a = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_a_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_a_ptr = tmp_buffer_a->ptr();

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpGEMM_workEstimation(handle,
                                                GetTransposeOperation(transa),
                                                GetTransposeOperation(transb),
                                                &alpha,
                                                a_descriptor.descriptor(),
                                                b_descriptor.descriptor(),
                                                &beta,
                                                out_descriptor.descriptor(),
                                                gpu_type,
                                                CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc,
                                                &buffer_a_size,
                                                tmp_buffer_a_ptr);
  });

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpGEMM_compute(handle,
                                         GetTransposeOperation(transa),
                                         GetTransposeOperation(transb),
                                         &alpha,
                                         a_descriptor.descriptor(),
                                         b_descriptor.descriptor(),
                                         &beta,
                                         out_descriptor.descriptor(),
                                         gpu_type,
                                         CUSPARSE_SPGEMM_DEFAULT,
                                         spgemmDesc,
                                         &buffer_b_size,
                                         nullptr);
  });

  phi::Allocator::AllocationPtr tmp_buffer_b = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_b_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_b_ptr = tmp_buffer_b->ptr();

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpGEMM_compute(handle,
                                         GetTransposeOperation(transa),
                                         GetTransposeOperation(transb),
                                         &alpha,
                                         a_descriptor.descriptor(),
                                         b_descriptor.descriptor(),
                                         &beta,
                                         out_descriptor.descriptor(),
                                         gpu_type,
                                         CUSPARSE_SPGEMM_DEFAULT,
                                         spgemmDesc,
                                         &buffer_b_size,
                                         tmp_buffer_b_ptr);
  });

  int64_t out_crows_size, out_cols_size, out_values_size;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMatGetSize(out_descriptor.descriptor(),
                                       &out_crows_size,
                                       &out_cols_size,
                                       &out_values_size);
  });

  // Reallocate space for cols and values of mat_out
  mat_out_cols->Resize(make_dim(out_values_size));
  dev_ctx_.template Alloc<int32_t>(mat_out_cols);
  mat_out_values->Resize(make_dim(out_values_size));
  dev_ctx_.template Alloc<T>(mat_out_values);

  phi::dynload::cusparseCsrSetPointers(
      out_descriptor.descriptor(),
      const_cast<int32_t*>(mat_out_crows->data<int32_t>()),
      const_cast<int32_t*>(mat_out_cols->data<int32_t>()),
      const_cast<T*>(mat_out_values->data<T>()));

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpGEMM_copy(handle,
                                      GetTransposeOperation(transa),
                                      GetTransposeOperation(transb),
                                      &alpha,
                                      a_descriptor.descriptor(),
                                      b_descriptor.descriptor(),
                                      &beta,
                                      out_descriptor.descriptor(),
                                      gpu_type,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc);
  });

  if (mat_a.crows().dtype() == phi::DataType::INT64 ||
      mat_b.crows().dtype() == phi::DataType::INT64) {
    phi::CastKernel<int32_t>(
        dev_ctx_, *mat_out_crows, phi::DataType::INT64, mat_out_crows);
    phi::CastKernel<int32_t>(
        dev_ctx_, *mat_out_cols, phi::DataType::INT64, mat_out_cols);
  }

  phi::dynload::cusparseSpGEMM_destroyDescr(spgemmDesc);
}
}  // namespace sparse
}  // namespace funcs
}  // namespace phi
