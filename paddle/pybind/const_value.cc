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

#include "const_value.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace pybind {

void BindConstValue(pybind11::module& m) {
  m.def("kEmptyVarName", [] { return framework::kEmptyVarName; });
  m.def("kTempVarName", [] { return framework::kTempVarName; });
  m.def("kGradVarSuffix", [] { return framework::kGradVarSuffix; });
  m.def("kZeroVarSuffix", [] { return framework::kZeroVarSuffix; });

  // for kernel_hint key
  m.def("kForceCPU", [] { return framework::kForceCPU; });
  m.def("kUseCUDNN", [] { return framework::kUseCUDNN; });
  m.def("kUseMKLDNN", [] { return framework::kUseMKLDNN; });
}

}  // namespace pybind
}  // namespace paddle
