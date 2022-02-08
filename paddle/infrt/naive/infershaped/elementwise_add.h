// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <llvm/ADT/SmallVector.h>

#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/naive/infershaped/infershaped_kernel_launcher.h"

// This file contains a example of the infershape ElementwiseAdd kernel.
// Some of the following code should be generated from PTEN by script.

namespace infrt {
namespace naive {

static void ElementwiseAddInferShape(const MetaTensor& a,
                                     const MetaTensor& b,
                                     MetaTensor* c) {
  CHECK(a.shape() == b.shape())
      << "ElementwiseAdd, but shapes of a b are not match";
  *c->mutable_shape() = a.shape();
}

static void ElementwiseAdd(const tensor::DenseHostTensor& a,
                           const tensor::DenseHostTensor& b,
                           tensor::DenseHostTensor* c) {}

// TODO(zhiqiang) This class should be generated by a script offline.
class ElementwiseAddLauncher : public InferShapedKernelLauncher {
 public:
  static const uint16_t input_tensor_indices[2];
  static const uint16_t num_input_tensors{2};
  static const bool turn_on_infer_shape_cache{true};

  void Invoke(host_context::KernelFrame* frame) override {
    // Build the infershape KernelFrame if needed.
    // TODO(Superjomn) add unlikely here.
    if (infershape_kernel_frame_builder.IsEmpty()) {
      CreateKernelFrameForInferShape(frame);
    }
    if (turn_on_infer_shape_cache) {
      if (IsShapeChanged(input_tensor_indices, num_input_tensors)) {
        INFRT_KERNEL(ElementwiseAddInferShape)
        (&infershape_kernel_frame_builder);
        BuildInferShapeCache(input_tensor_indices, num_input_tensors);
      }
    } else {
      INFRT_KERNEL(ElementwiseAddInferShape)(&infershape_kernel_frame_builder);
      BuildInferShapeCache(input_tensor_indices, num_input_tensors);
    }

    INFRT_KERNEL(ElementwiseAdd)(frame);
  }
};

const uint16_t ElementwiseAddLauncher::input_tensor_indices[2] = {0, 1};

}  // namespace naive
}  // namespace infrt
