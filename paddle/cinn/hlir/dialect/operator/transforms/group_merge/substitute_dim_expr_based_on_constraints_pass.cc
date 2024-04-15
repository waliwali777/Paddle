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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/substitute_dim_expr_based_on_constraints_pass.h"

#include "paddle/cinn/common/union_find.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

template <typename DoEachT>
void VisitEachOp(pir::Operation* op, const DoEachT& DoEach) {
  DoEach(op);
  for (auto& region : *op) {
    for (auto& block : region) {
      for (auto& op_in_block : block) {
        DoEach(&op_in_block);
      }
    }
  }
}

template <typename DoEachT>
void VisitEachValue(const pir::Operation* op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op->num_operands(); ++i) {
    DoEach(op->operand_source(i));
  }
  for (std::size_t i = 0; i < op->num_results(); ++i) {
    DoEach(op->result(i));
  }
}

std::unordered_map<symbol::DimExpr, symbol::DimExpr> GetDimExprSubstitution(
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  const std::vector<symbol::DimExprConstraint>& dim_expr_constraints =
      shape_analysis->DimExprBuilder().constraints();
  const cinn::common::UnionFindSet<symbol::DimExpr>& union_find_set = [&]() {
    cinn::common::UnionFindSet<symbol::DimExpr> union_find_set;
    for (const auto& constraint : dim_expr_constraints) {
      CHECK(std::holds_alternative<symbol::Equal<symbol::DimExpr>>(constraint))
          << "The DimExprConstraint type is no Equal<DimExpr>, this part is to "
             "be completed.";
      const auto& data =
          std::get<symbol::Equal<symbol::DimExpr>>(constraint).data;
      union_find_set.Union(data->lhs, data->rhs);
    }
    return union_find_set;
  }();

  const std::vector<std::vector<symbol::DimExpr>>& dim_expr_clusters =
      union_find_set.Clusters();
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitution_pattern;
  for (const auto& dim_expr_cluster : dim_expr_clusters) {
    CHECK(!dim_expr_cluster.empty());
    auto dim_expr_root = dim_expr_cluster[0];
    for (const auto& dim_expr : dim_expr_cluster) {
      if (symbol::GetDimExprPriority(dim_expr) <
          symbol::GetDimExprPriority(dim_expr_root)) {
        dim_expr_root = dim_expr;
      }
    }
    for (const auto& dim_expr : dim_expr_cluster) {
      if (dim_expr != dim_expr_root) {
        substitution_pattern[dim_expr] = dim_expr_root;
      }
    }
  }
  return substitution_pattern;
}

void SubstituteDimExprBasedOnConstraints(pir::Operation* region_op) {
  VLOG(4) << "SubstituteDimExprBasedOnConstraints start";
  pir::ShapeConstraintIRAnalysis* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(region_op->GetParentProgram());
  const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
      substitution_pattern = GetDimExprSubstitution(shape_analysis);

  VisitEachOp(region_op, [&](pir::Operation* op) {
    VisitEachValue(op, [&](pir::Value value) {
      if (!shape_analysis->HasShapeOrDataForValue(value)) {
        VLOG(4) << "Can not find ShapeOrData for value of op(" << op->name()
                << ") in shape_analysis";
      } else {
        const symbol::ShapeOrDataDimExprs& origin_shape_or_data =
            shape_analysis->GetShapeOrDataForValue(value);
        VLOG(8) << op->name()
                << "      origin_shape_or_data: " << origin_shape_or_data;
        const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
            symbol::SubstituteShapeOrData(origin_shape_or_data,
                                          substitution_pattern);
        VLOG(8) << op->name()
                << " substituted_shape_or_data: " << substituted_shape_or_data;
        shape_analysis->SetShapeOrDataForValue(value,
                                               substituted_shape_or_data);
      }
    });
    if (op->num_regions() > 0) {
      return;
    }
    if (op->num_results() > 0) {
      pir::shape::SetShapeAttrForOp(
          op, shape_analysis->GetShapeOrDataForValue(op->result(0)));
    } else {
      pir::shape::SetShapeAttrForOp(
          op, shape_analysis->GetShapeOrDataForValue(op->operand_source(0)));
    }
  });
  VLOG(4) << "SubstituteDimExprBasedOnConstraints end";
}

class SubstituteDimExprBasedOnConstraintsPass : public pir::Pass {
 public:
  SubstituteDimExprBasedOnConstraintsPass()
      : pir::Pass("substitute_dim_expr_based_on_constraints_pass", 1) {}

  void Run(pir::Operation* op) override {
    SubstituteDimExprBasedOnConstraints(op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateSubstituteDimExprBasedOnConstraintsPass() {
  return std::make_unique<SubstituteDimExprBasedOnConstraintsPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
