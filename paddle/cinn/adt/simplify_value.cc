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

#include "paddle/cinn/adt/simplify_value.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/match_and_rewrite.h"

namespace cinn::adt::equation {

struct SimplifyDotUndot {
  using source_pattern_type = IndexDot<IndexUnDot<Value>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    ADT_TODO();
  }
};

struct SimplifyUndotDot {
  using source_pattern_type = IndexUnDot<IndexDot<Value>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    ADT_TODO();
  }
};

struct SimplifyListGetItem {
  using source_pattern_type = ListGetItem<List<Value>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    ADT_TODO();
  }
};

// Only simplify top-layer of value
Value SimplifyValue(const IndexExprInferContext& ctx, Value value) {
  Value value = MatchAndRewrite<SimplifyDotUndot>(value, ctx);
  Value value = MatchAndRewrite<SimplifyUndotDot>(value, ctx);
  Value value = MatchAndRewrite<SimplifyListGetItem>(value, ctx);
  return value;
}

}  // namespace cinn::adt::equation
