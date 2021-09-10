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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/determinant_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;

template <typename T>
struct CudaMulFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] * args[1];
  }
};

template <typename T>
struct ElementwiseMulFunctor<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    std::vector<const framework::Tensor*> ins = {x, y};
    std::vector<framework::Tensor*> outs = {z};
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, -1, CudaMulFunctor<T>());
  }
};

template <typename T>
__global__ void DeterminantGrad(const size_t numel, T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numel) {
    out[tid] = static_cast<T>(1);
  }
}
template <typename T>
class DeterminantCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }

    auto rank = input_dim[input_dim_size - 1];
    DeterminantFunctor<T>()(*input, context, rank, batch_count, output);
    auto output_dims =
        framework::slice_ddim(input->dims(), 0, input_dim_size - 2);
    output->Resize(output_dims);
  }
};

template <typename T>
class DeterminantGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const T* dout_data = dout->data<T>();
    auto dout_dim = vectorize(dout->dims());

    auto* dx =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int64_t numel = dx->numel();
    for (int64_t idx = 0; idx < numel; idx++) {
      dx_data[idx] = static_cast<T>(1);
    }
  }
};

template <typename T>
class SlogDeterminantCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }

    auto rank = input_dim[input_dim_size - 1];
    SlogDeterminantFunctor<T>()(*input, context, rank, batch_count, output);
    std::vector<int> output_dim_vec(input_dim.begin(), input_dim.end() - 2);
    output_dim_vec.insert(output_dim_vec.begin(),
                          2);  // make the output dims as same as numpy
    auto output_dims = framework::make_ddim(output_dim_vec);
    output->Resize(output_dims);
    VLOG(2) << "output dim:" << output->dims();
  }
};

template <typename T>
class SlogDeterminantGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int threads = PADDLE_CUDA_NUM_THREADS;
    auto numel = output->numel() / 2;
    int blocks = (numel + threads - 1) / threads;

    auto rank = input_dim[input_dim_size - 1];
    DeterminantGrad<T><<<blocks, threads>>>(numel, output_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(determinant, ops::DeterminantCUDAKernel<float>,
                        ops::DeterminantCUDAKernel<double>);

REGISTER_OP_CUDA_KERNEL(
    determinant_grad,
    ops::DeterminantGradKernel<plat::CUDADeviceContext, float>,
    ops::DeterminantGradKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(slogdeterminant, ops::SlogDeterminantCUDAKernel<int>,
                        ops::SlogDeterminantCUDAKernel<int64_t>,
                        ops::SlogDeterminantCUDAKernel<float>,
                        ops::SlogDeterminantCUDAKernel<double>,
                        ops::SlogDeterminantCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(slogdeterminant_grad,
                        ops::SlogDeterminantGradCUDAKernel<float>,
                        ops::SlogDeterminantGradCUDAKernel<double>);
