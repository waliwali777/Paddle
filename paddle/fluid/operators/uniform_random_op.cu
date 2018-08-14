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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/uniform_random_op.h"

namespace paddle {
namespace operators {
template <typename T>
using GPUUniformRandomKernel =
    UniformRandomKernel<platform::CUDADeviceContext, T>;

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(uniform_random,
                        paddle::operators::GPUUniformRandomKernel<float>,
                        paddle::operators::GPUUniformRandomKernel<double>);
REGISTER_OP_CUDA_KERNEL(uniform_random_batch_size_like,
                        paddle::operators::GPUUniformRandomKernel<float>,
                        paddle::operators::GPUUniformRandomKernel<double>);
