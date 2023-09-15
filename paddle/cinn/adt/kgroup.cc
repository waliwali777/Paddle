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

#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/hlir/framework/graph.h"

namespace cinn::adt {

using AnchorTensor = Variable;

std::size_t GetTensorNumel(const Tensor& tensor) {
  CHECK(tensor.Has<adapter::Tensor>());
  return tensor.Get<adapter::Tensor>().GetNumel();
}

ScheduleDescriptor KGroup::GetDefaultScheduleDescriptor(
    const std::shared_ptr<IGroup>& igroup) const {
  const Tensor& tensor = igroup->anchor_tensor();

  CHECK_EQ(GetTensorNumel(tensor) % 64, 0);
  return {{S0x{}, GetTensorNumel(tensor) / 64}, {S1x{}, 64}};
}

}  // namespace cinn::adt
