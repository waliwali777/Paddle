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

#include "paddle/fluid/operators/cross_entropy_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;
using CUDACtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(cross_entropy,
                        ops::CrossEntropyOpKernel<CUDACtx, float>,
                        ops::CrossEntropyOpKernel<CUDACtx, double>,
                        ops::CrossEntropyOpKernel<CUDACtx, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    cross_entropy_grad, ops::CrossEntropyGradientOpKernel<CUDACtx, float>,
    ops::CrossEntropyGradientOpKernel<CUDACtx, double>,
    ops::CrossEntropyGradientOpKernel<CUDACtx, plat::float16>);

REGISTER_OP_CUDA_KERNEL(cross_entropy2,
                        ops::CrossEntropy2OpKernel<CUDACtx, float>,
                        ops::CrossEntropy2OpKernel<CUDACtx, double>,
                        ops::CrossEntropy2OpKernel<CUDACtx, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    cross_entropy2_grad, ops::CrossEntropy2GradientOpKernel<CUDACtx, float>,
    ops::CrossEntropy2GradientOpKernel<CUDACtx, double>,
    ops::CrossEntropy2GradientOpKernel<CUDACtx, plat::float16>);
