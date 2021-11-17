/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include "paddle/pten/core/backends/xpu/context.h"

namespace pten {

void XPUContext::Wait() const {
  int ret = xpu_set_device(place_.device);
  PADDLE_ENFORCE_EQ(ret,
                    XPU_SUCCESS,
                    paddle::platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  xpu_wait(context_->xpu_stream);
}

void XPUContext::SetContext(xpu::Context* context) { context_ = context; }

}  // namespace pten
