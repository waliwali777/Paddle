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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/tcmpt/core/storage.h"

namespace paddle {
namespace experimental {

class ExternalStorage : public tcmpt::Storage {
 public:
  ExternalStorage(void* ptr, size_t size, const platform::Place& place);
  ExternalStorage(const tcmpt::intrusive_ptr<tcmpt::Storage>& root,
                  size_t delta, size_t size);

  void Realloc(size_t n) override {
    PADDLE_THROW(platform::errors::Unavailable(
        "The external shared storage cannot be reallocated."));
  }

  size_t size() const noexcept override { return size_; }
  const platform::Place& place() const override { return data_.place(); }
  bool OwnsMemory() const noexcept override { return false; }

 private:
  const int64_t size_{0};
};

}  // namespace experimental
}  // namespace paddle
