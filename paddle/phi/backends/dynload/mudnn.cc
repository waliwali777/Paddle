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
#ifdef PADDLE_WITH_MUSA
#include "paddle/phi/backends/dynload/mudnn.h"

namespace phi {
namespace dynload {

bool HasCUDNN() {
  // note: mudnn.so is not imported by dlopen, which will be linked
  // in cmakelist.txt.
  return true;
}

void mudnnCreate(Handle** handle, int device) { *handle = new Handle(device); }

void mudnnSetStream(Handle* handle, musaStream_t stream) {
  handle->SetStream(stream);
}

void mudnnDestroy(Handle* handle) {
  if (handle != nullptr) {
    delete handle;
    handle = nullptr;
  }
}

}  // namespace dynload
}  // namespace phi
#endif