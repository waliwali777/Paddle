// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <vector>

namespace phi {
class DeviceContext;

namespace distributed {

class DistTensor;
class TensorDistAttr;

class ReshardFunction {
 public:
  ReshardFunction() = default;
  virtual ~ReshardFunction() = default;

  virtual bool IsSuitable(const DistTensor& in,
                          const TensorDistAttr& out_dist_attr) = 0;

  virtual DistTensor Eval(DeviceContext* dev_ctx,
                          const DistTensor& in,
                          const TensorDistAttr& out_dist_attr) = 0;
};

std::vector<std::unique_ptr<ReshardFunction>>& GetReshardFunctionList();

template <typename RESHARD_FUNC>
std::unique_ptr<RESHARD_FUNC> CreateReshardFunction() {
  return std::make_unique<RESHARD_FUNC>();
}

#define REGISTER_RESHARD_FUNC(func_type)       \
  class __RegisterReshard_##func_type {        \
   public:                                     \
    __RegisterReshard_##func_type() {          \
      GetReshardFunctionList().emplace_back(   \
          CreateReshardFunction<func_type>()); \
    }                                          \
  };                                           \
  static __RegisterReshard_##func_type local_reshard_func_##func_type

ReshardFunction* ChooseProperReshardFunction(
    const DistTensor& in, const TensorDistAttr& out_dist_attr);

}  // namespace distributed
}  // namespace phi
