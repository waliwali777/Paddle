/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fill_constant_op.h"
// #include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

template <typename T>
struct GaussianGenerator {
  T mean_, std_;
  unsigned int seed_;
  unsigned int offset_ = 0;

  __host__ __device__ GaussianGenerator(T mean, T std, int seed)
      : mean_(mean), std_(std), seed_(seed) {}

  __host__ __device__ GaussianGenerator(T mean, T std, int seed, int offset)
      : mean_(mean), std_(std), seed_(seed), offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::normal_distribution<T> dist(mean_, std_);
    unsigned int new_n = n + offset_;
    rng.discard(new_n);
    return dist(rng);
  }
};

template <typename T>
struct GaussianGeneratorOffset {
  T mean_, std_;
  unsigned int seed_;
  int offset_;

  __host__ __device__ GaussianGeneratorOffset(T mean, T std, int seed,
                                              int offset)
      : mean_(mean), std_(std), seed_(seed), offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::normal_distribution<T> dist(mean_, std_);
    rng.discard(n);
    return dist(rng);
  }
};

template <typename T>
class GPUGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor = context.Output<framework::Tensor>("Out");
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    bool seed_flag = false;
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
      seed_flag = true;
    }
    T mean = static_cast<T>(context.Attr<float>("mean"));
    T std = static_cast<T>(context.Attr<float>("std"));
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    const std::string op_type = "gaussian_random";
    auto shape = GetShape(context, op_type);
    tensor->Resize(shape);
    T* data = tensor->mutable_data<T>(context.GetPlace());

    int64_t size = tensor->numel();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(-1);

    if (gen_cuda->GetIsInitPy() && seed_flag) {
      std::cout << ">>>>>>>>CUDA GAUSSIAN GENERATOR" << std::endl;
      auto seed_offset = gen_cuda->IncrementOffset(1);
      // auto seed_gen = static_cast<unsigned int>(gen_cuda->GetCurrentSeed());
      // int offset_step = 0;
      // NOTE(xuefeng): Currently, we let offset step fixed to avoid
      // unexpected results which may cause ut fail.
      // we will fix this in future.
      // int gen_offset = offset_step * seed_offset.second;
      // std::cout << ">>>>>offset: " << gen_offset << std::endl;
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(data),
                        GaussianGenerator<T>(mean, std, seed_offset.first, 0));
    } else {
      std::cout << "COUNT ORIGIN" << std::endl;
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(data),
                        GaussianGenerator<T>(mean, std, seed));
    }
  }
};

template <typename T>
class GPUGaussianRandomBatchSizeLikeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    bool seed_flag = false;
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
      seed_flag = true;
    }
    T mean = static_cast<T>(context.Attr<float>("mean"));
    T std = static_cast<T>(context.Attr<float>("std"));
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int64_t size = tensor->numel();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(-1);

    if (gen_cuda->GetIsInitPy() && seed_flag) {
      std::cout << ">>>>>>>>CUDA GAUSSIAN GENERATOR" << std::endl;
      // auto seed_offset = gen_cuda->IncrementOffset(1);
      auto seed_gen = static_cast<unsigned int>(gen_cuda->GetCurrentSeed());
      // int offset_step = 0;
      // NOTE(xuefeng): Currently, we let offset step fixed to avoid
      // unexpected results which may cause ut fail.
      // we will fix this in future.
      // int gen_offset = offset_step * seed_offset.second;
      // std::cout << ">>>>>offset: " << gen_offset << std::endl;
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(data),
                        GaussianGenerator<T>(mean, std, seed_gen, 0));
    } else {
      std::cout << "COUNT ORIGIN" << std::endl;
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(data),
                        GaussianGenerator<T>(mean, std, seed));
    }
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(gaussian_random,
                        paddle::operators::GPUGaussianRandomKernel<float>,
                        paddle::operators::GPUGaussianRandomKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    gaussian_random_batch_size_like,
    paddle::operators::GPUGaussianRandomBatchSizeLikeKernel<float>,
    paddle::operators::GPUGaussianRandomBatchSizeLikeKernel<double>);
