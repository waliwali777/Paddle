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

#pragma once
#include <string>

#include "paddle/phi/common/pstring.h"
#include "paddle/phi/kernels/strings/unicode.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/transform.h"

namespace phi {
namespace strings {

using pstring = dtype::pstring;
struct AsciiToLower {
  HOSTDEVICE char operator()(char in) const {
    return ('A' <= in && in <= 'Z') ? in - ('Z' - 'z') : in;
  }
};

struct AsciiToUpper {
  HOSTDEVICE char operator()(char in) const {
    return ('a' <= in && in <= 'z') ? in ^ 0x20 : in;
  }
};

template <typename DeviceContext>
struct UTF8ToLower {
  HOSTDEVICE UTF8ToLower(uint8_t* unicode_flag_map, uint16_t* cases_map)
      : unicode_flag_map_(unicode_flag_map), cases_map_(cases_map) {}

  HOSTDEVICE uint32_t operator()(uint32_t in) const {
    uint32_t flg = (in <= 0x00FFFF ? unicode_flag_map_[in] : 0);
    return (strings::isupper(flg) ? cases_map_[in] : in);
  }

  uint8_t* unicode_flag_map_;
  uint16_t* cases_map_;
};

template <typename DeviceContext>
struct UTF8ToUpper {
  HOSTDEVICE UTF8ToUpper(uint8_t* unicode_flag_map, uint16_t* cases_map)
      : unicode_flag_map_(unicode_flag_map), cases_map_(cases_map) {}

  HOSTDEVICE uint32_t operator()(uint32_t in) const {
    uint32_t flg = (in <= 0x00FFFF ? unicode_flag_map_[in] : 0);
    return (strings::islower(flg) ? cases_map_[in] : in);
  }

  uint8_t* unicode_flag_map_;
  uint16_t* cases_map_;
};

}  // namespace strings
}  // namespace phi
