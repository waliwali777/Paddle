// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/group_cluster/common_utils.h"

namespace cinn::frontend::group_cluster::policy {

struct ShardableAxes {
  ShardableAxes() : axis_names({}) {}
  explicit ShardableAxes(const std::vector<std::string>& names)
      : axis_names(names) {}
  std::vector<std::string> axis_names;
  std::string DebugStr();
};

struct ShardableAxesSignature {
  std::vector<ShardableAxes> inputs;
  std::vector<ShardableAxes> outputs;
  std::string DebugStr();
};

struct ShardableAxesInfoManager {
  ShardableAxesInfoManager(
      const std::vector<const pir::Operation*>& ops,
      const pir::ShapeConstraintIRAnalysis* shape_analysis);
  ShardableAxesSignature GetSignature(const pir::Operation* op);
  ShardableAxes GetAxes(pir::Value value);
  ShardableAxesSignature CreateShardableSignature(const pir::Operation* op);
  ShardableAxes ReplaceShardableAxesWithRootName(const ShardableAxes& axes);
  static std::string GetUniqueName();

 private:
  const std::vector<const pir::Operation*>& ops_;
  const pir::ShapeConstraintIRAnalysis* shape_analysis_;

  std::unordered_map<const pir::Operation*, ShardableAxesSignature>
      op_signature_map_;
  std::unordered_map<pir::Value, ShardableAxes> value_axes_map_;
  std::unordered_map<std::string, std::string> name_union_;
};

}  // namespace cinn::frontend::group_cluster::policy
