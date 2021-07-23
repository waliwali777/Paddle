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
#pragma once

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/module/GlobalFunctor.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/fast_divmod.h"

#ifdef __HIPCC__
#define ELEMENTWISE_BLOCK_SIZE 256
#else
#define ELEMENTWISE_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

enum ElementwiseType { kUnary = 1, kBinary = 2 };

/*
* According to NVIDIA, if number of threads per block is 64/128/256/512,
* cuda performs better. And number of blocks should be greater (at least
* 2x~4x) than number of SMs. Hence, SM count is took into account within
* this function to determine the right number of threads per block.
*/
inline int GetThreadsConfig(const platform::CUDADeviceContext &ctx,
                            int64_t numel, int vec_size) {
  int threads = ELEMENTWISE_BLOCK_SIZE;
  int sm_count = ctx.GetSMCount();
  int active_threads_num = numel / vec_size;
  if (active_threads_num / (sm_count << 1) < ELEMENTWISE_BLOCK_SIZE) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about twice of SM, to acquire better performance.
    threads = platform::RoundToPowerOfTwo(active_threads_num / (sm_count << 1));
  } else if (active_threads_num / (sm_count << 2) < ELEMENTWISE_BLOCK_SIZE) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about 4 times of SM, to acquire better performance.
    threads = platform::RoundToPowerOfTwo(active_threads_num / (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  return std::max(64, threads);
}

/*
* Only the address of input data is the multiplier of 1,2,4, vectorized load
* with corresponding multiplier-value is possible. Moreover, the maximum length
* of vectorized load is 128 bits once. Hence, valid length of vectorized load
* shall be determined under both former constraints.
*/
template <typename T>
int GetVectorizedSizeImpl(const T *pointer) {
  constexpr int max_load_bits = 128;
  int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec8 =
      std::alignment_of<CudaAlignedVector<T, 8>>::value;  // NOLINT
  constexpr int vec4 =
      std::alignment_of<CudaAlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec2 =
      std::alignment_of<CudaAlignedVector<T, 2>>::value;  // NOLINT
  if (address % vec8 == 0) {
    /*
    * Currently, decide to deal with no more than 4 data once while adopting
    * vectorization load/store, if performance test shows that dealing with
    * 8 data once in vectorization load/store does get optimized, return code
    * below can be changed into " return std::min(8, valid_vec_size); " .
    */
    return std::min(4, valid_vec_size);
  } else if (address % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

template <typename InT, typename OutT>
int GetVectorizedSize(const std::vector<const framework::Tensor *> &ins,
                      const std::vector<framework::Tensor *> &outs) {
  int vec_size = 4;
  for (auto iter = ins.begin(); iter != ins.end(); ++iter) {
    vec_size =
        std::min<int>(vec_size, GetVectorizedSizeImpl((*iter)->data<InT>()));
  }
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size =
        std::min<int>(vec_size, GetVectorizedSizeImpl((*iter)->data<OutT>()));
  }
  return vec_size;
}

template <ElementwiseType ET, int VecSize, typename InT, typename OutT,
          typename Functor>
__device__ inline void Compute(const InT *__restrict__ in0,
                               const InT *__restrict__ in1, OutT *out,
                               Functor func) {
  using InVecType = CudaAlignedVector<InT, VecSize>;
  using OutVecType = CudaAlignedVector<OutT, VecSize>;
  const InVecType *src1 = reinterpret_cast<const InVecType *>(in0);
  const InVecType *src2 = reinterpret_cast<const InVecType *>(in1);
  OutVecType *dst = reinterpret_cast<OutVecType *>(out);
  InVecType arg1;
  InVecType arg2;
  InT args[ET][VecSize];
  modules::read_data<InVecType, 1, 1, 1>(src1, &arg1, 0);
  modules::read_data<InVecType, 1, 1, 1>(src2, &arg2, 0);
  modules::vectype_2_type<InVecType, InT, VecSize>(arg1, &args[0][0]);
  modules::vectype_2_type<InVecType, InT, VecSize>(arg2, &args[1][0]);
  InT data[ET];
  OutVecType result;
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
#pragma unroll
    for (int j = 0; j < ET; ++j) {
      data[j] = args[j][i];
    }
    result.val[i] = static_cast<OutT>(func(data));
  }
  modules::write_data<OutVecType, 1, 1, 1>(&result, dst, 0);
}

template <ElementwiseType ET, int VecSize, typename InT, typename OutT,
          typename Functor>
__global__ void ElementVectorized(const InT *__restrict__ in0,
                                  const InT *__restrict__ in1, OutT *out,
                                  int size, Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int fix = VecSize * tid;
  int remain = size - fix;
  remain = remain > 0 ? remain : 0;
  if (remain >= VecSize) {
    // VectorizedKernelImpl(data, func, tid);
    Compute<ET, VecSize, InT, OutT, Functor>(in0 + fix, in1 + fix, out + fix,
                                             func);
  } else {
    for (int i = 0; i < remain; i++) {
      Compute<ET, 1, InT, OutT, Functor>(in0 + fix + i, in1 + fix + i,
                                         out + fix + i, func);
    }
  }
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
__global__ void ElementScalar(const InT *__restrict__ in0,
                              const InT *__restrict__ in1, OutT *out, int size,
                              Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    Compute<ET, 1, InT, OutT, Functor>(in0 + tid, in1 + tid, out + tid, func);
  }
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchSameDimsElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, Functor func) {
  // calculate the max vec_size for all ins and outs
  auto size = ins[0]->numel();
  int vec_size = GetVectorizedSize<InT, OutT>(ins, *outs);
  int block_size = GetThreadsConfig(ctx, size, vec_size);
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  const InT *in0 = ins[0]->data<InT>();
  const InT *in1 =
      (ET == ElementwiseType::kBinary) ? ins[1]->data<InT>() : nullptr;
  OutT *out = (*outs)[0]->data<OutT>();
  // cuda kernel
  auto stream = ctx.stream();

  switch (vec_size) {
    case 4:
      ElementVectorized<ET, 4><<<grid_size, block_size, 0, stream>>>(
          in0, in1, out, size, func);
      break;
    case 2:
      ElementVectorized<ET, 2><<<grid_size, block_size, 0, stream>>>(
          in0, in1, out, size, func);
      break;
    case 1:
      ElementScalar<ET><<<grid_size, block_size, 0, stream>>>(in0, in1, out,
                                                              size, func);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
  }
}

}  // namespace operators
}  // namespace paddle
