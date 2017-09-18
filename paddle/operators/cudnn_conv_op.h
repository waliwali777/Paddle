/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/operators/conv_base.h"

namespace paddle {
namespace operators {

// If CudnnConvOp is running on CPU use the implementation
// from ConvBaseKernel.
template <typename Place, typename T>
class CudnnConvKernel : ConvBaseKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

template <typename Place, typename T>
class CudnnConvGradKernel : ConvBaseGradKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

}  // namespace operators
}  // namespace paddle
