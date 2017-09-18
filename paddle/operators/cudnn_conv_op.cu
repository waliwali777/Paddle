/* Copyright (c) 2016 PaddlePaddle Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

template <typename T>
class CudnnConvOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dialations = ctx.Attr<std::vector<int>>("dialations");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>();

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNHWC;

    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, input->dims());
    cudnnTensorDescriptor_t cudnn_output_desc =
        output_desc.descriptor<T>(layout, output->dims());
    cudnnFilterDescriptor_t cudnn_filter_desc =
        filter_desc.descriptor(layout, filter->dims());
    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor(paddings, strides, dialations);
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo;
    PADDLE_ENFORCE(cudnnGetConvolutionForwardAlgorithm(
        ctx.device_context()->cudnn_handle(), cudnn_input_desc,
        cudnn_filter_desc, cudnn_conv_desc, cudnn_output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    // ------------------- cudnn conv workspace ---------------------
    void* cudnn_workspace = NULL;
    size_t workspace_size_in_bytes = 0;
    PADDLE_ENFORCE(cudnnGetConvolutionForwardWorkspaceSize(
        ctx.device_context()->cudnn_handle(), cudnn_input_desc,
        cudnn_filter_desc, cudnn_conv_desc, cudnn_output_desc, algo,
        &size_in_bytes));
    cudaMalloc(&cudnn_workspace, workspace_size_in_bytes);
    // ------------------- cudnn conv forward ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    platform::DataLayout layout = platform::DataLayout::kNCHW float alpha =
        1.0f,
                         beta = 0.0f;
    PADDLE_ENFORCE(cudnnConvolutionForward(
        ctx.device_context()->cudnn_handle(), &alpha, cudnn_input_desc,
        input_data, cudnn_filter_desc, filter_data, cudnn_conv_desc, algo,
        cudnn_workspace, workspace_size_in_bytes, &beta, cudnn_output_desc,
        output_data));
  }
};

template <typename T>
class CudnnConvGradOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    // input is actually the output of forward.
    auto input = ctx.Input<Tensor>("Input");
    // filter is the filter used in forward.
    auto filter = ctx.Input<Tensor>("Filter");
    // output_grad is gradient output of the operator connected after it.
    auto output_grad = ctx.Input<Tensor>("Output");
    // input_grad is the gradient output of conv op.
    auto input_grad = ctx.Output<Tensor>("Input");
    // filter_grad is the gradient output of filter(kernel).
    auto filter_grad = ctx.Output<Tensor>("Filter");

    const T* input_data = input->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data = input_grad->mutable_data<T>();
    T* filter_grad_data = filter_grad->mutable_data<T>();

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_grad_desc;
    ScopedTensorDescriptor input_grad_desc;
    ScopedTensorDescriptor filter_grad_desc;

    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNHWC;

    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, input->dims());
    cudnnTensorDescriptor_t cudnn_output_grad_desc =
        output_grad_desc.descriptor<T>(layout, output_grad->dims());
    cudnnFilterDescriptor_t cudnn_filter_desc =
        filter_desc.descriptor(layout, filter->dims());
    cudnnFilterDescriptor_t cudnn_input_grad_desc =
        input_grad_desc.descriptor(layout, input_grad->dims());
    cudnnFilterDescriptor_t cudnn_filter_grad_desc =
        filter_grad_desc.descriptor(layout, filter_grad->dims());

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor(paddings, strides, dialations);
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t data_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t workspace_size_in_bytes = 0, tmp_size = 0;

    PADDLE_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm(
        ctx.device_context()->cudnn_handle(), cudnn_filter_desc,
        // dyDesc: Handle to the previously initialized input differential
        // tensor descriptor.
        cudnn_output_grad_desc, cudnn_conv_desc,
        // dxDesc: Handle to the previously initialized output tensor
        // descriptor.
        cudnn_input_grad_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
        &data_algo));
    PADDLE_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(
        ctx.device_context()->cudnn_handle(), cudnn_filter_desc,
        cudnn_output_grad_desc, cudnn_conv_desc, cudnn_input_grad_desc,
        data_algo, &tmp_size));
    workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);

    PADDLE_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm(
        ctx.device_context()->cudnn_handle(), cudnn_input_desc,
        cudnn_input_grad_desc, cudnn_conv_desc, cudnn_filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &filter_algo));

    PADDLE_ENFORCE(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        ctx.device_context()->cudnn_handle(), cudnn_input_desc,
        cudnn_input_grad_desc, cudnn_conv_desc, cudnn_filter_desc, filter_algo,
        &tmp_size));
    workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size)

        // ------------------- cudnn conv workspace ---------------------
        void* cudnn_workspace = NULL;
    cudaMalloc(&cudnn_workspace, workspace_size_in_bytes);
    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    platform::DataLayout layout = platform::DataLayout::kNCHW float alpha =
        1.0f,
                         beta = 0.0f;
    PADDLE_ENFORCE(cudnnConvolutionBackwardData(
        ctx.device_context()->cudnn_handle(), &alpha, cudnn_filter_desc,
        filter_data, cudnn_output_grad_desc, output_grad_data, cudnn_conv_desc,
        data_algo, cudnn_workspace, workspace_size_in_bytes, &beta,
        cudnn_input_grad_desc, input_grad_data));
    // ------------------- cudnn conv backward filter ---------------------
    PADDLE_ENFORCE(cudnnConvolutionBackwardFilter(
        ctx.device_context()->cudnn_handle(), &alpha, cudnn_input_desc,
        input_data, cudnn_output_grad_desc, output_grad_data, cudnn_conv_desc,
        filter_algo, cudnn_workspace, workspace_size_in_bytes, &beta,
        cudnn_filter_grad_desc, filter_grad_data));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_GPU_KERNEL(cudnn_conv, paddle::operators::CudnnConvOpKernel<float>);
REGISTER_OP_GPU_KERNEL(cudnn_conv_grad,
                       paddle::operators::CudnnConvGradOpKernel<float>);
