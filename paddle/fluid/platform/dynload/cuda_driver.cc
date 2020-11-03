/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/dynload/rocm_driver.h"
#endif  

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag cuda_dso_flag;
void* cuda_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUDA_ROUTINE_EACH(DEFINE_WRAP);

bool HasCUDADriver() {
  std::call_once(cuda_dso_flag, []() { cuda_dso_handle = GetCUDADsoHandle(); });
  return cuda_dso_handle != nullptr;
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
