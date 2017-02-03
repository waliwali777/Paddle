/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <memory>
#include "paddle/utils/Flags.h"

namespace paddle {

//! GradientMachineAttributes
struct GradientMachineAttributes {
  //! GradientMachine uses Model Parallel or not. It should be set by
  //! FLAGS_parallel_nn.
  bool parallelNeuralNetowrk;

  inline bool useGPU(int deviceID) const {
    return parallelNeuralNetowrk ? (deviceID >= 0 ? true : false)
                                 : FLAGS_use_gpu;
  }

  //! TODO(yuayng18): Add more flags here.
};

typedef std::shared_ptr<GradientMachineAttributes> GradientMachineAttrPtr;

}  // namespace paddle
