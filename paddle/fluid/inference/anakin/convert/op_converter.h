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

#include <string>
#include <vector>
#include "framework/core/types.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "saber/saber_types.h"

namespace paddle {
namespace inference {
namespace anakin {

class OpConverter {
 public:
  OpConverter() {}

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::Scope &scope) = 0;
  void ConvertOp(const framework::proto::OpDesc &op,
                 const std::unordered__set<std::string> &parameters,
                 const framework::Scope &scope, AnakinEngine *engine) {
    framework::OpDesc op_desc(op, nullptr);
    OpConverter *it = nullptr;
  }

  void ConvertBlock(const framework::proto::BlockDesc &block,
                    const std::unordered_set<std::string> &parameters,
                    const framework::Scope &scope, AnakinEngine *engine) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto i = 0; i < block.op_size(); i++) {
      auto &op = block.ops(i);
      ConvertOp(op, parameters, scope, engine);
    }
  }
  void SetEngine(AnakinEngine *engine) { engine_ = engine; }
  virtual OpConverter() {}
  AnakinEngine *engine_{nullptr};

 private:
  std::unordered_map<std::string, OpConvert *> converters_;
  framework::Scope *scope_{nullptr};
  std::mutex mutex_;
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
