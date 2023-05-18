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

#include "paddle/ir/program.h"

#include <unordered_map>
#include <vector>

#include "paddle/ir/ir_context.h"
#include "paddle/ir/value.h"

namespace ir {
Program::~Program() {
  enum Mark { Unvisited, Temp, Perm };
  std::unordered_map<void*, int> visited;
  for (auto op : ops_) {
    visited[static_cast<void*>(op)] = Mark::Unvisited;
  }

  std::vector<Operation*> topology_order_ops;
  topology_order_ops.reserve(ops_.size());

  std::function<void(Operation*)> dfs;

  dfs = [&](Operation* op) -> void {
    auto* key = static_cast<void*>(op);
    if (visited[key] == Mark::Perm) return;
    if (visited[key] == Mark::Temp) throw("has cycle");

    visited[key] = Mark::Temp;
    for (size_t idx = 0; idx < op->num_results(); idx++) {
      auto ret = op->GetResultByIndex(idx);
      for (auto iter = ret.begin(); iter != ret.end(); ++iter) {
        dfs(iter.owner());
      }
    }

    visited[key] = Mark::Perm;
    topology_order_ops.push_back(op);
  };
  if (topology_order_ops.size() > 0) {
    dfs(topology_order_ops[0]);
  }
  std::reverse(topology_order_ops.begin(), topology_order_ops.end());
  for (auto op : topology_order_ops) {
    op->destroy();
  }
}

void Program::InsertOp(Operation* op) {
  ops_.push_back(op);
  op->set_parent_program(this);
}

Parameter* Program::GetParameter(std::string name) const {
  if (parameters_.count(name) != 0) {
    return parameters_.at(name).get();
  }
  return nullptr;
}

void Program::SetParameter(std::string name,
                           std::unique_ptr<Parameter>&& parameter) {
  parameters_[name].reset(parameter.release());
}

}  // namespace ir
