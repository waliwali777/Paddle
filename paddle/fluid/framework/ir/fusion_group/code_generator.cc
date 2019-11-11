/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/code_generator.h"
#include <set>
#include <sstream>
#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"
#include "paddle/fluid/framework/ir/fusion_group/operation.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

CodeGenerator::CodeGenerator() {
  // Only support elementwise operations now.
  code_templates_.resize(1);

  CodeTemplate elementwise_t(elementwise_cuda_template);
  code_templates_[0] = elementwise_t;
}

std::string CodeGenerator::Generate(SubGraph* subgraph) {
  std::unordered_map<std::string, int> var_ids = EncodeVarNodes(subgraph);
  std::vector<OperationExpression> expressions;
  for (auto* node : subgraph->SortedNodes()) {
    if (node && node->IsOp() && node->Op()) {
      auto* op = node->Op();
      LOG(INFO) << "expression for " << op->Type();

      // Input ids should be set in fixed order, like:
      //  - x, y in forward operations
      //  - x, y, out, out@GRAD in backward operations
      std::vector<int> input_ids;
      std::vector<std::string> input_names =
          OperationMap::Instance().Get(op->Type()).input_names;
      for (auto& name : input_names) {
        // TODO(liuyiqun): support duplicated input.
        if (op->Input(name).size() >= 1U) {
          // Some input vars are not used in grad ops, such as
          // "elementwise_add_grad", where "X", "Y" and "Out" are not used.
          PADDLE_ENFORCE_NE(var_ids.find(op->Input(name)[0]), var_ids.end(),
                            "Input(%s) of operation %s should be set.", name,
                            op->Type());
          input_ids.push_back(var_ids[op->Input(name)[0]]);
        } else {
          input_ids.push_back(-1);
        }
      }
      // Output ids should be set in fixed order, like:
      //  - dx, dy in backward operations
      std::vector<int> output_ids;
      std::vector<std::string> output_names =
          OperationMap::Instance().Get(op->Type()).output_names;
      for (auto& name : output_names) {
        PADDLE_ENFORCE_EQ(op->Output(name).size(), 1U,
                          "Output(%s) of operation %s should be set.", name,
                          op->Type());
        PADDLE_ENFORCE_NE(var_ids.find(op->Output(name)[0]), var_ids.end(),
                          "Output(%s) of operation %s should be set.", name,
                          op->Type());
        output_ids.push_back(var_ids[op->Output(name)[0]]);
      }
      InsertOperationExpression(
          &expressions,
          OperationExpression(node->Name(), input_ids, output_ids));
    }
  }
  return Generate(subgraph->func_name, expressions);
}

// In order to get the right result of expression, we need to calculate and
// store the expression as suffix Expressions using vector.
std::string CodeGenerator::Generate(
    std::string func_name, std::vector<OperationExpression> expressions) {
  // Check whether all expressions are elementwise operations.
  TemplateVariable template_var;
  template_var.Add("func_name", func_name);
  template_var.Add("parameters", EmitParameters(expressions, "float"));
  template_var.Add("compute_body", EmitComputeBody(expressions));
  return predefined_cuda_functions + code_templates_[0].Format(template_var);
}

// we get the parameter list code for the expression information
std::string CodeGenerator::EmitParameters(
    std::vector<OperationExpression> expressions, std::string dtype) {
  std::set<int> input_ids;
  std::set<int> output_ids;
  // Remove the reptead id and get a ordered list.
  for (size_t i = 0; i < expressions.size(); i++) {
    for (auto id : expressions[i].GetInputIds()) {
      input_ids.insert(id);
    }
    for (auto id : expressions[i].GetOutputIds()) {
      output_ids.insert(id);
    }
  }

  // If a id is in the input and output list at the same time, then remove it
  // from the input list.
  for (auto iter = input_ids.begin(); iter != input_ids.end();) {
    if (output_ids.find(*iter) != output_ids.end()) {
      input_ids.erase(iter++);
    } else {
      iter++;
    }
  }

  std::stringstream ret;
  ret << "int N, ";
  for (auto iter = input_ids.begin(); iter != input_ids.end(); iter++) {
    ret << dtype << "* " << VarName(*iter) << ", ";
  }

  size_t count_index = 0;
  for (auto iter = output_ids.begin(); iter != output_ids.end(); iter++) {
    ret << dtype << "* " << VarName(*iter);
    if (count_index != output_ids.size() - 1) {
      ret << ", ";
    }
    count_index++;
  }

  return ret.str();
}

std::string CodeGenerator::EmitComputeBody(
    std::vector<OperationExpression> expressions) {
  // get the right experssion code using suffix expression
  std::stringstream ret;
  for (size_t i = 0; i < expressions.size(); i++) {
    LOG(INFO) << DebugString(expressions[i]);
    ret << expressions[i].GetExpression();
  }
  return ret.str();
}

std::unordered_map<std::string, int> CodeGenerator::EncodeVarNodes(
    SubGraph* subgraph) {
  const auto& input_var_nodes = subgraph->GetInputVarNodes();
  const auto& output_var_nodes = subgraph->GetOutputVarNodes();

  int id = 0;
  std::unordered_map<std::string, int> var_ids;
  // Numbering input vars.
  for (auto* in : input_var_nodes) {
    LOG(INFO) << "input names:" << in->Name() << ", id:" << id;
    if (var_ids.find(in->Name()) == var_ids.end()) {
      var_ids[in->Name()] = id++;
    }
  }
  // Numbering internal vars.
  for (auto* node : subgraph->SortedNodes()) {
    if (node && node->IsVar() && node->Var()) {
      bool is_found = false;
      for (auto* in : input_var_nodes) {
        if (node == in) {
          is_found = true;
          break;
        }
      }
      if (is_found) {
        continue;
      }
      for (auto* out : output_var_nodes) {
        if (node == out) {
          is_found = true;
          break;
        }
      }
      PADDLE_ENFORCE_EQ(
          is_found, true,
          "Subgraph with internal var nodes (%s) is not supported yet.",
          node->Name());
    }
  }
  // Encoding output vars.
  for (auto* out : output_var_nodes) {
    LOG(INFO) << "output names:" << out->Name() << ", id:" << id;
    if (var_ids.find(out->Name()) == var_ids.end()) {
      var_ids[out->Name()] = id++;
    }
  }
  return var_ids;
}

void CodeGenerator::InsertOperationExpression(
    std::vector<OperationExpression>* expressions, OperationExpression expr) {
  if (expressions->size() == 0U) {
    expressions->push_back(expr);
  } else {
    int from = 0;
    int to = expressions->size();
    for (int i = 0; i < static_cast<int>(expressions->size()); ++i) {
      // Check whether i-expr's output is expr's input.
      for (auto in : expr.GetInputIds()) {
        for (auto out : expressions->at(i).GetOutputIds()) {
          // Insert expr after i-expr.
          if (in == out && i >= from) {
            from = i + 1;
          }
        }
      }
      // Check whether i-expr's input is expr's output.
      for (auto out : expr.GetOutputIds()) {
        for (auto in : expressions->at(i).GetInputIds()) {
          // Insert expr before i-expr.
          if (out == in && to > i) {
            to = i;
          }
        }
      }
    }
    PADDLE_ENFORCE_LE(from, to, "Range [%d, %d] is invalid");
    LOG(INFO) << "from:" << from << ", to:" << to;
    expressions->insert(expressions->begin() + from, expr);
  }
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
