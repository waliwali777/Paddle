/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file defines the the class to partition a graph.
 */

#pragma once
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace inference {
namespace analysis {
using framework::ir::Node;

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes);

void RenameAndGetOutputs(
    const std::vector<framework::ir::Node *> &subgraph_nodes,
    framework::BlockDesc *block_desc,
    const std::set<std::string> &input_names_with_id,
    std::set<std::string> *output_names_with_id,
    std::set<std::string> *output_names,
    std::unordered_map<std::string, std::string> *output_name_map,
    const std::unordered_map<std::string, framework::ir::Node *> &graph_var_map,
    bool is_trt = true);

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
