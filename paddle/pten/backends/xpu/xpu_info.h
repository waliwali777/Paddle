/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

// #ifdef PADDLE_WITH_XPU

#include <string>
#include <vector>
#include "paddle/pten/common/place.h"

namespace pten {

class XPUContext;

namespace backends {
namespace xpu {

/***** Version Management *****/

//! Get the version of XPU Driver
int GetDriverVersion();

//! Get the version of XPU Runtime
int GetRuntimeVersion();

/***** Device Management *****/

//! Get the total number of XPU devices in system.
int GetXPUDeviceCount(const std::string &);

//! Set the XPU device id for next execution.
void SetXPUDeviceId(int device_id,
                    const std::string &xpu_visible_devices_str = "");

//! Get the current XPU device id in system.
int GetXPUCurrentDeviceId();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices(
    const std::string &selected_xpus = "",
    const std::string &xpu_visible_devices_str = "");

/***** Memory Management *****/

//! Copy memory from address src to dst synchronously.
void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const pten::XPUPlace &dst_place);
void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const pten::XPUPlace &src_place,
                   const pten::XPUContext &dev_ctx);
void MemcpySyncD2D(void *dst,
                   const pten::XPUPlace &dst_place,
                   const void *src,
                   const pten::XPUPlace &src_place,
                   size_t count,
                   const pten::XPUContext &dev_ctx);

class XPUDeviceGuard {
 public:
  explicit inline XPUDeviceGuard(int dev_id) {
    int prev_id = GetXPUCurrentDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      SetXPUDeviceId(dev_id);
    }
  }

  inline ~XPUDeviceGuard() {
    if (prev_id_ != -1) {
      SetXPUDeviceId(prev_id_);
    }
  }

  XPUDeviceGuard(const XPUDeviceGuard &o) = delete;
  XPUDeviceGuard &operator=(const XPUDeviceGuard &o) = delete;

 private:
  int prev_id_{-1};
};

enum XPUVersion { XPU1, XPU2 };
XPUVersion get_xpu_version(int dev_id);

}  // namespace xpu
}  // namespace backends
}  // namespace pten

// #endif
