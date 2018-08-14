// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/random/identity_distribution.h"
#include "paddle/fluid/platform/random/normal_distribution.h"
#include "paddle/fluid/platform/random/philox_engine.h"
#include "paddle/fluid/platform/random/random_sequence.h"
#include "paddle/fluid/platform/random/uniform_distribution.h"

namespace paddle {
namespace platform {

using random::IdentityDistribution;
using random::NormalDistribution;
using random::Philox32x4;
using random::RandomFill;
using random::RandomSequence;
using random::UniformRealDistribution;
using DefaultRandomEngine = random::Philox32x4;

}  // namespace platform
}  // namespace paddle
