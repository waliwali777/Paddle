/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"
#include <boost/lexical_cast.hpp>

namespace paddle {
namespace framework {

void OperatorBase::Init(const OpDesc &op_desc, AttributeMap& attrs) {
  desc_ = op_desc;
  inputs_.reserve(desc_.inputs_size());
  for (auto& input : desc_.inputs()) {
    inputs_.push_back(input);
  }
  outputs_.reserve(desc_.outputs_size());
  for (auto& output : desc_.outputs()) {
    outputs_.push_back(output);
  }
  for(auto it = attrs.begin(); it != attrs.end(); ++it) {
    attrs_[it->first] = it->second;
  }
}

Variable* OperatorBase::Input(Scope* scope, int index) const {
  return scope->GetVariable(inputs_[index]);
}

Variable* OperatorBase::Output(Scope* scope, int index) const {
  return scope->GetVariable(outputs_[index]);
}

Attribute OperatorBase::GetAttr(std::string name) {
  return attrs_[name];
}

void OperatorBase::InferShape(Scope *scope) const {}

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "=================\n";
  ss << "type = " << type() << "\n";
  ss << "inputs = [";
  for (auto& ipt : inputs_) {
    ss << ipt << ", ";
  }
  ss << "]\n";
  ss << "outputs = [";
  for (auto& opt : outputs_) {
    ss << opt << ", ";
  }
  ss << "]\n";
  ss << "attr_keys = [";
  for (auto& attr : attrs_) {
    ss << attr.first << ", ";
  }
  ss << "]\n";
  return ss.str();
}

}  // namespace framework
}  // namespace paddle