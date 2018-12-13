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

#include "paddle/fluid/platform/temporay_allocator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"

namespace paddle {
namespace platform {
namespace alloc = memory::allocation;

TemporayAllocation::TemporayAllocation(
    alloc::AllocationPtr &&underlying_allocation)
    : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)) {}

TemporayAllocator::TemporayAllocator(platform::Place place) : place_(place) {
  mtx_.reset(new std::mutex());
  temp_allocations_.reset(new std::deque<TemporayAllocation *>());
}

bool TemporayAllocator::IsAllocThreadSafe() const { return true; }

void TemporayAllocator::Release() {
  std::shared_ptr<std::deque<TemporayAllocation *>> t_allocations;
  {
    platform::LockGuardPtr<std::mutex> guard(mtx_);
    t_allocations = temp_allocations_;
    temp_allocations_.reset(new std::deque<TemporayAllocation *>());
  }
  for (auto tmp : *t_allocations) {
    delete tmp;
  }
}
void TemporayAllocator::Free(alloc::Allocation *allocation) {
  auto temp_allocation = dynamic_cast<TemporayAllocation *>(allocation);
  PADDLE_ENFORCE_NOT_NULL(temp_allocation);

  if (platform::is_gpu_place(temp_allocation->place())) {
    platform::LockGuardPtr<std::mutex> guard(mtx_);
    temp_allocations_->emplace_back(temp_allocation);
    return;
  }
  delete temp_allocation;
}

alloc::Allocation *TemporayAllocator::AllocateImpl(
    size_t size, alloc::Allocator::Attr attr) {
  auto raw_allocation =
      alloc::AllocatorFacade::Instance().Alloc(place_, size, attr);
  return new TemporayAllocation(std::move(raw_allocation));
}

}  // namespace platform
}  // namespace paddle
