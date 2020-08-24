// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/isfinite_v2_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_OVERFLOW_CUDA_KERNEL(op_type, functor)                       \
  REGISTER_OP_CUDA_KERNEL(                                                    \
      op_type, ops::OverflowKernel<paddle::platform::CUDADeviceContext, int,  \
                                   ops::functor>,                             \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, int64_t,       \
                          ops::functor>,                                      \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, float,         \
                          ops::functor>,                                      \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, double,        \
                          ops::functor>,                                      \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, plat::float16, \
                          ops::functor>);

REGISTER_OVERFLOW_CUDA_KERNEL(isinf_v2, InfinityV2Functor);
REGISTER_OVERFLOW_CUDA_KERNEL(isnan_v2, NANV2Functor);
REGISTER_OVERFLOW_CUDA_KERNEL(isfinite_v2, IsfiniteV2Functor);
