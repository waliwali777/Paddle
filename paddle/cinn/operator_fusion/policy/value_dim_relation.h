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
#include <functional>
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

struct ValueDim {
  ValueDim(const pir::Value& v, const size_t idx, const size_t usage_idx)
      : v_(v), idx_(idx), usage_idx_(usage_idx) {}
  pir::Value v_;
  size_t idx_;
  size_t usage_idx_;  // value is used by which op
  ValueDim(pir::Value v, size_t idx) : v_(v), idx_(idx) {}
  ValueDim() = default;
  ValueDim(const ValueDim& v) = default;
  bool operator==(const ValueDim& v) const {
    return (idx_ == v.idx_) && (v_ == v.v_) && (usage_idx_ == v.usage_idx_);
  }

  size_t GetNumericValue() const {
    return v_.type().dyn_cast<pir::DenseTensorType>().dims().at(idx_);
  }

  std::string DebugStr() const {
    std::ostringstream oss;
    oss << "ValueDim UsageIdx " << usage_idx_ << " : ";
    oss << "Index: " << idx_;
    oss << ", ";
    v_.defining_op()->Print(oss);
    return oss.str();
  }
};

static std::size_t hash_two(std::size_t h1, std::size_t h2) {
  return h1 ^ (h2 << 1);
}

struct ValueDimHash {
  std::size_t operator()(const ValueDim& p) const {
    auto h1 = std::hash<size_t>{}(p.idx_);
    auto h2 = std::hash<pir::Value>{}(p.v_);
    auto h3 = std::hash<size_t>{}(p.usage_idx_);
    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return hash_two(hash_two(h1, h2), h3);
  }
};

using ValueDimRelation =
    std::unordered_map<ValueDim,
                       std::unordered_map<ValueDim, bool, ValueDimHash>,
                       ValueDimHash>;
// ValueDimRelation[in][out] = True; means f(out) = in is related.

ValueDimRelation AnalysisIndexExprRelation(
    const std::vector<pir::Operation*>& ops);
const size_t GetUsageIdx(const pir::Value& v, pir::Operation* op);
std::vector<ValueDim> GetAllValueDimFromValue(const pir::Value& v,
                                              const size_t usage_idx);

}  // namespace cinn::fusion
