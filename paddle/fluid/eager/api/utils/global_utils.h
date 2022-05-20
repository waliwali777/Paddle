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
//

#pragma once

#include <atomic>
#include <memory>
#include "paddle/fluid/eager/type_defs.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/utils/small_vector.h"

namespace egr {
class UniqueNameGenerator {
 public:
  explicit UniqueNameGenerator(std::string prefix = "") : prefix_(prefix) {}
  std::string Generate(std::string key = "eager_tmp") {
    return prefix_ + key + "_" + std::to_string(id_++);
  }

 private:
  std::atomic<int> id_{0};
  std::string prefix_;
};

// Global
// TODO(jiabin): Now we are using imperative tracer, move it here when we
// deprecate imperative.

class Controller {
 public:
  static Controller& Instance() { return *controller_; }
  paddle::platform::Place GetExpectedPlace() const {
    return tracer_->ExpectedPlace();
  }
  void SetExpectedPlace(const paddle::platform::Place& place) {
    tracer_->SetExpectedPlace(place);
  }
  void SetAMPLevel(paddle::imperative::AmpLevel level) {
    tracer_->SetAmpLevel(level);
  }
  paddle::imperative::AmpLevel GetAMPLevel() const {
    return tracer_->GetAmpLevel();
  }
  bool HasGrad() const { return tracer_->HasGrad(); }
  void SetHasGrad(bool has_grad) { tracer_->SetHasGrad(has_grad); }
  std::string GenerateUniqueName(std::string key = "eager_tmp") {
    return tracer_->GenerateUniqueName(key);
  }
  const std::shared_ptr<paddle::imperative::Tracer>& GetCurrentTracer() {
    return tracer_;
  }
  void SetCurrentTracer(
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    tracer_ = tracer;
    VLOG(6) << "Set current tracer for Controller: " << tracer_;
  }

  const std::unordered_map<std::string, std::vector<paddle::OpMetaInfo>>&
  GetOpMetaInfoMap() {
    return op_meta_info_map_;
  }

  void MergeOpMetaInfoMap(const std::unordered_map<
                          std::string, std::vector<paddle::OpMetaInfo>>& map) {
    op_meta_info_map_.insert(map.begin(), map.end());
  }

  std::unordered_map<std::string,
                     std::vector<std::vector<std::unordered_map<int, int>>>>&
  GetCustomEdgesSlotMap() {
    return custom_edges_slot_map_;
  }

 private:
  Controller() = default;
  static Controller* controller_;
  std::shared_ptr<paddle::imperative::Tracer> tracer_{
      new paddle::imperative::Tracer()};
  std::unordered_map<std::string, std::vector<paddle::OpMetaInfo>>
      op_meta_info_map_;
  /* op_type : {{{grad_outputs}, {grad_inputs}, {input}, {output}, {attrs}},
   * {{grad_outputs}, {grad_inputs}, {input}, {output}, {attrs}}}*/
  std::unordered_map<std::string,
                     std::vector<std::vector<std::unordered_map<int, int>>>>
      custom_edges_slot_map_;
  DISABLE_COPY_AND_ASSIGN(Controller);
};

}  // namespace egr
