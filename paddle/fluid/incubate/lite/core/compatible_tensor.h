// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "tensor.h"

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#include "paddle/fluid/incubate/lite/core/lite_tensor.h"
#else
#include "hvy_tensor.h"
#endif

namespace paddle {
namespace lite {

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
using DDim = lite::DDimLite;
using Tensor = lite::TensorLite;
#else
using DDim = lite::DDimHvy;
using Tensor = lite::TensorHvy;
#endif

}  // namespace lite
}  // namespace paddle
