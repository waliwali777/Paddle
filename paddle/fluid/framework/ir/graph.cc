/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {

std::unique_ptr<Graph> ProgramToGraph(const ProgramDesc &program) {
  std::unique_ptr<Graph> graph(new Graph(program));

  std::unordered_map<std::string, VarDesc *> all_vars;
  for (auto *var : program.Block(0).AllVars()) {
    all_vars.emplace(var->Name(), var);
  }

  for (auto *op : program.Block(0).AllOps()) {
    ir::Node *node = graph->CreateOpNode(op);

    for (auto &each_var_name : op->InputArgumentNames()) {
      ir::Node *var = nullptr;
      if (all_vars.count(each_var_name) != 0) {
        var = graph->CreateVarNode(all_vars.at(each_var_name));
      } else {
        LOG(ERROR) << "input var not in all_var list: " << each_var_name;
        var = graph->CreateEmptyNode(each_var_name);
      }
      node->inputs.push_back(var);
      var->outputs.push_back(node);
    }

    for (auto &each_var_name : op->OutputArgumentNames()) {
      ir::Node *var = graph->CreateVarNode(all_vars.at(each_var_name));
      node->outputs.push_back(var);
      var->inputs.push_back(node);
    }
  }
  return std::move(graph);
}

}  // namespace framework
}  // namespace paddle
