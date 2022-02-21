/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF NCHW KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/kernels/copy_kernel.h"
#include "paddle/pten/kernels/sparse/convolution_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"

namespace pten {
namespace tests {

std::vector<int> flatten(const std::vector<std::vector<int>>& in) {
  std::vector<int> out;
  if (in.size() == 0) return out;
  const int rows = in.size();
  const int cols = in[0].size();
  const int n = rows * cols;
  out.resize(n);
  for (int i = 0; i < rows; i++) {
    memcpy(&out[i * cols], in[i].data(), cols * sizeof(int));
  }
  return out;
}

template <typename T>
void TestConv3d(const std::vector<int>& indices,
                const std::vector<T>& features,
                const DDim& x_dims,
                const std::vector<T>& kernel,
                const DDim& kernel_dims,
                const std::vector<int>& correct_out_indices,
                const std::vector<T>& correct_out_features,
                const DDim& correct_out_dims,
                const int non_zero_num,
                const std::vector<int>& paddings,
                const std::vector<int>& strides,
                const std::vector<int>& dilations) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  pten::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  pten::CPUPlace cpu;

  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  DenseTensor indices_tensor(
      alloc.get(),
      DenseTensorMeta(DataType::INT32, {4, non_zero_num}, DataLayout::NCHW));
  memcpy(indices_tensor.mutable_data<int>(cpu),
         indices.data(),
         indices.size() * sizeof(int));
  DenseTensor features_tensor(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {non_zero_num, in_channels},
                      DataLayout::NCHW));
  memcpy(features_tensor.mutable_data<T>(cpu),
         features.data(),
         features.size() * sizeof(T));

  SparseCooTensor x_tensor(indices_tensor, features_tensor, x_dims);

  // TODO(zhangkaihuo) change layout to DHWCOC
  DenseTensor kernel_tensor(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      kernel_dims,
                      DataLayout::NCHW));
  memcpy(kernel_tensor.mutable_data<T>(cpu),
         kernel.data(),
         kernel.size() * sizeof(T));

  const float diff = 1e-1;
  if (false) {
    SparseCooTensor out = sparse::Conv3d<T>(dev_ctx_cpu,
                                            x_tensor,
                                            kernel_tensor,
                                            paddings,
                                            "valid",
                                            dilations,
                                            strides,
                                            1);

    ASSERT_EQ(correct_out_dims.size(), out.dims().size());
    ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, out.nnz());
    for (int i = 0; i < correct_out_dims.size(); i++) {
      ASSERT_EQ(correct_out_dims[i], out.dims()[i]);
    }

    int cmp_indices = memcmp(correct_out_indices.data(),
                             out.non_zero_indices().data<int>(),
                             correct_out_indices.size() * sizeof(int));
    ASSERT_EQ(cmp_indices, 0);

    for (uint64_t i = 0; i < correct_out_features.size(); i++) {
      float tmp = std::fabs(static_cast<float>(
          correct_out_features[i] - out.non_zero_elements().data<T>()[i]));
      ASSERT_LT(tmp, diff);
    }
  }
// test gpu
#if defined(PADDLE_WITH_CUDA)
  pten::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(pten::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();

  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_indices_tensor(
      cuda_alloc.get(),
      DenseTensorMeta(DataType::INT32, {4, non_zero_num}, DataLayout::NCHW));
  pten::Copy(dev_ctx_gpu, indices_tensor, true, &d_indices_tensor);
  DenseTensor d_features_tensor(
      cuda_alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {non_zero_num, in_channels},
                      DataLayout::NCHW));
  pten::Copy(dev_ctx_gpu, features_tensor, true, &d_features_tensor);

  SparseCooTensor d_x_tensor(d_indices_tensor, d_features_tensor, x_dims);

  // TODO(zhangkaihuo) change layout to DHWCOC
  DenseTensor d_kernel_tensor(
      cuda_alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      kernel_dims,
                      DataLayout::NCHW));
  pten::Copy(dev_ctx_gpu, kernel_tensor, true, &d_kernel_tensor);

  SparseCooTensor d_out = sparse::Conv3d<T>(dev_ctx_gpu,
                                            d_x_tensor,
                                            d_kernel_tensor,
                                            paddings,
                                            "valid",
                                            dilations,
                                            strides,
                                            1);

  for (int i = 0; i < 10; i++) {
    sparse::Conv3d<T>(dev_ctx_gpu,
                      d_x_tensor,
                      d_kernel_tensor,
                      paddings,
                      "valid",
                      dilations,
                      strides,
                      1);
  }

  ASSERT_EQ(correct_out_dims.size(), d_out.dims().size());
  ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, d_out.nnz());
  for (int i = 0; i < correct_out_dims.size(); i++) {
    ASSERT_EQ(correct_out_dims[i], d_out.dims()[i]);
  }

  DenseTensor h_indices_tensor(
      alloc.get(),
      DenseTensorMeta(DataType::INT32, {4, d_out.nnz()}, DataLayout::NCHW));
  pten::Copy(dev_ctx_gpu, d_out.non_zero_indices(), true, &h_indices_tensor);

  int cmp_indices2 = memcmp(correct_out_indices.data(),
                            h_indices_tensor.data<int>(),
                            correct_out_indices.size() * sizeof(int));
  ASSERT_EQ(cmp_indices2, 0);

  DenseTensor h_features_tensor(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {d_out.nnz()},
                      d_out.layout()));
  pten::Copy(dev_ctx_gpu, d_out.non_zero_elements(), true, &h_features_tensor);
  for (uint64_t i = 0; i < correct_out_features.size(); i++) {
    float tmp = std::fabs(static_cast<float>(correct_out_features[i] -
                                             h_features_tensor.data<T>()[i]));
    ASSERT_LT(tmp, diff);
  }
#endif
}

TEST(DEV_API, sparse_conv3d) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 4, 4, 4, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 2, 2, 2, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 4;
  // const int64_t sparse_dim = 4;
  std::vector<std::vector<int>> indices = {// {0, 0, 3, 3},
                                           // {0, 2, 2, 2},
                                           // {0, 0, 2, 3},
                                           // {0, 2, 3, 2}
                                           {0, 0, 0, 0},
                                           {0, 2, 0, 2},
                                           {3, 2, 2, 3},
                                           {3, 2, 3, 2}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {-0.2883, 0.0287, 0.2864, -0.0992};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.4721, 0.2292, 0.9751, 0.8616, 0.5784, 0.9178, 0.8727, 0.1659, 0.4455,

      0.0189, 0.4646, 0.4472, 0.1991, 0.8968, 0.3717, 0.0051, 0.6963, 0.2690,

      0.7473, 0.5403, 0.5391, 0.0796, 0.4734, 0.9097, 0.1712, 0.6237, 0.8837};

  std::vector<std::vector<int>> out_indices = {// {0, 0, 0, 0},
                                               // {0, 0, 0, 1},
                                               // {0, 0, 1, 0},
                                               // {0, 0, 1, 1},
                                               // {0, 1, 0, 0},
                                               // {0, 1, 0, 1},
                                               // {0, 1, 1, 0},
                                               // {0, 1, 1, 1}
                                               {0, 0, 0, 0, 0, 0, 0, 0},
                                               {0, 0, 0, 0, 1, 1, 1, 1},
                                               {0, 0, 1, 1, 0, 0, 1, 1},
                                               {0, 1, 0, 1, 0, 1, 0, 1}};
  std::vector<int> out_indices_flatten = flatten(out_indices);

  std::vector<float> out_features = {
      0.0254, 0.1455, -0.0615, 0.0862, 0.0077, 0.0200, -0.0160, -0.0433};

  if (false) {
    TestConv3d<float>(indices_flatten,
                      features,
                      x_dims,
                      kernel,
                      kernel_dims,
                      out_indices_flatten,
                      out_features,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
  }
}

TEST(DEV_API, sparse_conv3d_batch) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {2, 4, 4, 4, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {2, 2, 2, 2, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 8;
  // const int64_t sparse_dim = 4;
  std::vector<std::vector<int>> indices = {// {0, 0, 3, 3},
                                           // {0, 2, 2, 2},
                                           // {0, 0, 2, 3},
                                           // {0, 2, 3, 2}
                                           {0, 0, 0, 0, 1, 1, 1, 1},
                                           {0, 2, 0, 2, 0, 2, 0, 2},
                                           {3, 2, 2, 3, 3, 2, 2, 3},
                                           {3, 2, 3, 2, 3, 2, 3, 2}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {
      -0.2883, 0.0287, 0.2864, -0.0992, -0.2883, 0.0287, 0.2864, -0.0992};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.4721, 0.2292, 0.9751, 0.8616, 0.5784, 0.9178, 0.8727, 0.1659, 0.4455,

      0.0189, 0.4646, 0.4472, 0.1991, 0.8968, 0.3717, 0.0051, 0.6963, 0.2690,

      0.7473, 0.5403, 0.5391, 0.0796, 0.4734, 0.9097, 0.1712, 0.6237, 0.8837};

  std::vector<std::vector<int>> out_indices = {
      // {0, 0, 0, 0},
      // {0, 0, 0, 1},
      // {0, 0, 1, 0},
      // {0, 0, 1, 1},
      // {0, 1, 0, 0},
      // {0, 1, 0, 1},
      // {0, 1, 1, 0},
      // {0, 1, 1, 1}
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
      {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
      {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},
      {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};
  std::vector<int> out_indices_flatten = flatten(out_indices);

  std::vector<float> out_features = {0.0254,
                                     0.1455,
                                     -0.0615,
                                     0.0862,
                                     0.0077,
                                     0.0200,
                                     -0.0160,
                                     -0.0433,
                                     0.0254,
                                     0.1455,
                                     -0.0615,
                                     0.0862,
                                     0.0077,
                                     0.0200,
                                     -0.0160,
                                     -0.0433};

  if (false) {
    TestConv3d<float>(indices_flatten,
                      features,
                      x_dims,
                      kernel,
                      kernel_dims,
                      out_indices_flatten,
                      out_features,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
  }
}

template <typename T, typename Functor>
void LoadData(const std::string& filename,
              std::vector<T>* data,
              const int n,
              const Functor& load) {
  FILE* fp = fopen(filename.c_str(), "r");
  printf("%s\n", filename.c_str());
  if (fp == NULL) {
    return;
  }
  int i = 0;
  for (i = 0; i < n && fp; i++) {
    load(fp, &(*data)[i]);
  }
  if (i != n) {
    printf("%s\n", filename.c_str());
  }
  fclose(fp);
}

TEST(DEV_API, performance) {
  const int in_channels = 19;
  const int out_channels = 17;
  DDim x_dims = {2, 400, 400, 15, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {2, 398, 398, 13, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 60000;
  const int out_non_zero_num = 1187206;
  std::vector<int> indices(non_zero_num * 4), out_indices(out_non_zero_num * 4);
  using pten::dtype::float16;
  std::vector<float16> features(non_zero_num * in_channels),
      kernels(product(kernel_dims)),
      out_features(out_non_zero_num * out_channels);

  auto load_int = [](FILE* fp, int* data) { fscanf(fp, "%d\n", data); };
  // auto load_float = [](FILE* fp, float* data) { fscanf(fp, "%f\n", data); };
  auto load_fp16 = [](FILE* fp, pten::dtype::float16* data) {
    float tmp;
    fscanf(fp, "%f\n", &tmp);
    *data = static_cast<pten::dtype::float16>(tmp);
  };

  const std::string base_path = "/home/zhangkaihuo/project/Paddle/build/";
  LoadData(base_path + "indices.txt", &indices, non_zero_num * 4, load_int);
  LoadData(base_path + "out_indices.txt",
           &out_indices,
           out_non_zero_num * 4,
           load_int);
  LoadData(base_path + "features.txt",
           &features,
           non_zero_num * in_channels,
           load_fp16);
  LoadData(
      base_path + "kernels.txt", &kernels, product(kernel_dims), load_fp16);
  LoadData(base_path + "out_features.txt",
           &out_features,
           out_non_zero_num * out_channels,
           load_fp16);

  TestConv3d<pten::dtype::float16>(indices,
                                   features,
                                   x_dims,
                                   kernels,
                                   kernel_dims,
                                   out_indices,
                                   out_features,
                                   out_dims,
                                   non_zero_num,
                                   paddings,
                                   strides,
                                   dilations);
}

}  // namespace tests
}  // namespace pten
