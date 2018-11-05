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

#include "paddle/fluid/framework/details/analysis_var_pass.h"
#include <iostream>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace framework {
namespace details {

using details::UnlivedNodePool;
using details::ReusedNodePairMap;
using details::ControlFlowGraph;

template <typename Container>
std::string PrintIt(const AnalysisVarPass* pass, const Container& cons) {
  std::stringstream ss;
  for (auto& item : cons) {
    ss << pass->DebugString(item) << " ";
  }
  ss << std::endl;
  return ss.str();
}

const std::string AnalysisVarPass::DebugString(ir::Node* var) const {
  std::stringstream ss;
  ss << var->Name();
  ss << "[";
  auto shape = var->Var()->GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != shape.size() - 1) {
      ss << shape[i] << ",";
    } else {
      ss << shape[i];
    }
  }
  ss << "]";
  return ss.str();
}

bool AnalysisVarPass::NodeMatch(ir::Node* var, ir::Node** cache,
                                int* node_idx_in_pool) const {
  auto get_node_size = [&](ir::Node* n) {
    auto* desc = n->Var();
    auto shape = desc->GetShape();
    size_t type_size =
        framework::SizeOfType(framework::ToTypeIndex(desc->GetDataType()));
    return type_size * std::abs(std::accumulate(shape.begin(), shape.end(), 1));
  };

  auto compare_node_size = [&](ir::Node* lhs, ir::Node* rhs) {
    auto* lhs_desc = lhs->Var();
    auto* rhs_desc = rhs->Var();
    auto lhs_shape = lhs_desc->GetShape();
    auto rhs_shape = rhs_desc->GetShape();

    // -1 means shape is determined in runtime.
    if ((lhs_shape[0] == -1) ^ (rhs_shape[0] == -1)) return false;
    return get_node_size(lhs) <= get_node_size(rhs);
  };

  // linear search in an sorted node set, find the best fit node.
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    if (compare_node_size(var, *it)) {
      *cache = *it;
      *node_idx_in_pool = std::distance(pool.begin(), it);
      return true;
    }
  }
  return false;
}

std::unordered_set<ir::Node*> AnalysisVarPass::GetSubBlockOutputVars(
    const std::set<ir::Node*>& nodes) const {
  std::unordered_set<ir::Node*> vars;
  std::unordered_map<std::string, ir::Node*> var_to_node_map;
  ProgramDesc* program = nullptr;
  for (auto& op : nodes) {
    if (op->IsOp() && program == nullptr) {
      program = op->Op()->Block()->Program();
    }
    if (op->IsVar() && !op->IsCtrlVar()) {
      var_to_node_map[op->Name()] = op;
    }
  }
  if (program->Size() > 1) {
    // size>1 means program has subblock. A subblock's AllVars
    // only contains the output variables, no input variables
    // are included.
    for (size_t i = 1; i < program->Size(); ++i) {
      auto& block_desc = program->Block(i);
      auto subblock_output_vars = block_desc.AllVars();
      for (auto& var_desc : subblock_output_vars) {
        vars.insert(var_to_node_map[var_desc->Name()]);
      }
    }
  }
  return vars;
}

std::unique_ptr<ir::Graph> AnalysisVarPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& node_pair_map = Get<ReusedNodePairMap>(kGlobalReusedNodePairMap);
  auto& graph_ops = Get<std::vector<ir::Node*>>(kGraphReusedOps);
  auto nodes = graph->Nodes();
  auto subblock_output_vars = GetSubBlockOutputVars(nodes);

  ControlFlowGraph cfg(*graph.get());
  cfg.LiveVariableAnalysis();
  auto op_has_subblock = [&](OpDesc* desc) {
    const AttributeMap& attrs = desc->GetAttrMap();
    for (auto& attr : attrs) {
      if (typeid(attr.second) == typeid(BlockDesc)) return true;
    }
    return false;
  };

  auto var_can_reused = [&](ir::Node* node) {
    PADDLE_ENFORCE(
        node->IsVar() && !node->IsCtrlVar(),
        string::Sprintf("Expect node %s as Variable.", node->Name()));
    auto* desc = node->Var();
    proto::VarType::Type type = desc->GetType();
    if (desc->Persistable() || (type != proto::VarType::LOD_TENSOR &&
                                type != proto::VarType::SELECTED_ROWS) ||
        desc->GetShape().size() == 0) {
      return false;
    }
    if (subblock_output_vars.count(node)) return false;
    for (auto* op : node->inputs) {
      // fill_constant on cpu can not be shared.
      if (op->Name() == "fill_constant" && op->Op()->HasAttr("force_cpu")) {
        return framework::AttrReader(op->Op()->GetAttrMap())
                   .Get<bool>("force_cpu") != true;
      }
    }
    return true;
  };

  for (auto& op : cfg.Ops()) {
    VLOG(3) << PrintIt(this, pool);
    VLOG(3) << op->Name();
    auto* op_desc = op->Op();
    if (op_has_subblock(op_desc)) {
      VLOG(3) << op->Name() << " has subblock, skipped.";
      continue;
    }
    graph_ops.push_back(op);

    for (auto& var : cfg.Def(op)) {
      VLOG(3) << "start var " << DebugString(var);
      if (var_can_reused(var)) {
        int node_idx_in_pool = -1;
        ir::Node* cached_var = nullptr;
        VLOG(3) << "match var " << DebugString(var);
        // VLOG(3) << "result " << NodeMatch(var, &cached_var,
        // &node_idx_in_pool);
        if (NodeMatch(var, &cached_var, &node_idx_in_pool)) {
          VLOG(3) << string::Sprintf(
              "Hit Cache !!! cache pool index %d, var is %s, cached var %s",
              node_idx_in_pool, DebugString(var), DebugString(cached_var));
          // VLOG(3) << "matched " << DebugString(var);
          cfg.UpdateGraph(var, cached_var, node_idx_in_pool);
          // VLOG(3) << "update graph";
          pool.erase(cached_var);
          node_pair_map[op] = std::make_pair(var, cached_var);
        }
        // VLOG(3) << "match finish";
      }
    }

    for (auto var : cfg.LiveIn(op)) {
      if (var_can_reused(var) && cfg.LiveOut(op).count(var) == 0 &&
          pool.count(var) == 0) {
        pool.insert(var);
      }
    }
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(analysis_var_pass, paddle::framework::details::AnalysisVarPass)
    .RequirePassAttr(paddle::framework::details::kGlobalReusedNodePairMap)
    .RequirePassAttr(paddle::framework::details::kGraphReusedOps);
