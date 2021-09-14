// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/device_event.h"

namespace paddle {
namespace framework {

class StreamAnalyzer {
 public:
  explicit StreamAnalyzer(const platform::Place& place)
      : place_(place), d2h_ctx_pool_({place}), h2d_ctx_pool_({place}) {}

  ~StreamAnalyzer() {}

  void Schedule(const std::vector<OpFuncNode>& op_func_nodes,
                const std::vector<size_t>& downstream_ops, size_t op_index,
                std::vector<Instruction>* instructions);

  platform::DeviceContext* ParseDeviceContext(const OpFuncNode& op_func_node,
                                              const OperatorBase& op_base);

 private:
  std::vector<size_t> ParseEventVarIds(const Instruction& cur_instr,
                                       const Instruction& next_instr);

  void AssociateInputWithEvents(const std::vector<size_t>& new_event_var_id,
                                Instruction* next_instr, bool is_sync);
  platform::Place place_;
  platform::DeviceContextPool d2h_ctx_pool_;
  platform::DeviceContextPool h2d_ctx_pool_;
  std::map<size_t, std::shared_ptr<platform::DeviceEvent>> var_id2event_;
};

}  // namespace framework
}  // namespace paddle
