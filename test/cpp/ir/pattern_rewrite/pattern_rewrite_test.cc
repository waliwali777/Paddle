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

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/transforms/transform_general_functions.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/cast_utils.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/parameter.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/ir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/ir/pattern_rewrite/pattern_applicator.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"
#include "paddle/ir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/ir/transforms/dce.h"

// NOTE(zhangbo9674): File pd_op.h is generated by op_gen.py, see details in
// paddle/fluid/ir/dialect/CMakeLists.txt.
#include "paddle/fluid/ir/dialect/pd_op.h"

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/phi/core/ddim.h"

// Define op1.
class Operation1 : public ir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.Operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];
  void Verify();
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};

void Operation1::Verify() {
  auto &attributes = this->attributes();
  if (attributes.count("op2_attr1") == 0 ||
      (!attributes.at("op2_attr1").isa<ir::StrAttribute>())) {
    throw("Type of attribute: parameter_name is not right.");
  }
  if (attributes.count("op2_attr2") == 0 ||
      (!attributes.at("op2_attr2").isa<ir::StrAttribute>())) {
    throw("Type of attribute: parameter_name is not right.");
  }
}
const char *Operation1::attributes_name[attributes_num] = {"op2_attr1",
                                                           "op2_attr2"};
IR_DECLARE_EXPLICIT_TYPE_ID(Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(Operation1)

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

 private:
  void initialize() { RegisterOps<Operation1>(); }
};
IR_DECLARE_EXPLICIT_TYPE_ID(TestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(TestDialect)

// TODO(wilber): Add logical when ir support erase, replace or update.
class TestPatternRewrite : public ir::OpRewritePattern<Operation1> {
 public:
  using ir::OpRewritePattern<Operation1>::OpRewritePattern;

  void Rewrite(Operation1 op, ir::PatternRewriter &rewriter) const override {}
  bool Match(Operation1 op) const override { return false; }
};

class TestPatternRewrite2 : public ir::OpRewritePattern<Operation1> {
 public:
  using ir::OpRewritePattern<Operation1>::OpRewritePattern;
  bool MatchAndRewrite(
      Operation1 op,
      ir::PatternRewriter &rewriter) const override {  // NOLINT
    return false;
  }
};

TEST(PatternRewrite, PatternBenefit) {
  ir::PatternBenefit benefit1(1);
  EXPECT_EQ(benefit1.benefit(), 1U);
  ir::PatternBenefit benefit2(2);
  EXPECT_EQ(benefit2.benefit(), 2U);

  EXPECT_TRUE(benefit2 > benefit1);
  EXPECT_TRUE(benefit2 >= benefit1);
  EXPECT_TRUE(benefit1 < benefit2);
  EXPECT_TRUE(benefit1 <= benefit2);
  EXPECT_TRUE(benefit1 != benefit2);
  ir::PatternBenefit benefit3(2);
  EXPECT_TRUE(benefit2 == benefit3);
}

TEST(RewritePattern, RewritePatternSet) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();

  ir::RewritePatternSet ps(ctx);
  ps.Add<TestPatternRewrite>(ctx, 1);
  EXPECT_EQ(ps.native_patterns().size(), 1U);
  EXPECT_TRUE(ps.native_patterns().back()->debug_labels().empty());
  EXPECT_EQ(ps.native_patterns().back()->benefit(), 1U);
  ps.AddWithLabel<TestPatternRewrite2>({"TestPatternRewrite2"}, ctx, 2);
  EXPECT_EQ(ps.native_patterns().size(), 2U);
  EXPECT_EQ(ps.native_patterns().back()->debug_labels()[0],
            "TestPatternRewrite2");
  EXPECT_EQ(ps.native_patterns().back()->benefit(), 2U);

  ps.Clear();
  ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);
  EXPECT_EQ(ps.native_patterns().size(), 2U);
  EXPECT_EQ(ps.native_patterns()[0]->benefit(), 2U);
  EXPECT_EQ(ps.native_patterns()[1]->benefit(), 2U);
}

// TODO(wilber): Add actual case.
// TEST(PatternRewrite, PatternApplicator) {
//   ir::IrContext *ctx = ir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
//   auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
//   test_dialect->RegisterOp<Operation1>();
//   ir::RewritePatternSet ps(ctx);
//   ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);
//   ir::FrozenRewritePatternSet frozen_set(std::move(ps));
//   ir::PatternApplicator applicator(frozen_set);
//   applicator.ApplyDefaultCostModel();
// }

// // TODO(wilber): Add actual case.
TEST(PatternRewrite, FrozenRewritePatternSet) {
  ir::FrozenRewritePatternSet frozen_set;
  EXPECT_TRUE(frozen_set.match_any_op_native_patterns().empty());
  EXPECT_TRUE(frozen_set.op_specific_native_patterns().empty());

  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();
  ir::RewritePatternSet ps(ctx);
  ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);

  ir::FrozenRewritePatternSet frozen_set2(std::move(ps));
  EXPECT_TRUE(frozen_set2.match_any_op_native_patterns().empty());
  const auto &pattern_maps = frozen_set2.op_specific_native_patterns();
  EXPECT_EQ(pattern_maps.size(), 1U);
  EXPECT_EQ(pattern_maps.at(ctx->GetRegisteredOpInfo("test.Operation1")).size(),
            2U);
}

class TransposePatternRewrite
    : public ir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using ir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       ir::PatternRewriter &rewriter) const override {
    auto prev_op = ir::GetDefiningOpForInput<0>(op);
    std::vector<int> axis_last = GetAxis(op);
    auto prev_trans_op = prev_op->dyn_cast<paddle::dialect::TransposeOp>();
    if (prev_trans_op) {
      std::vector<int> axis_first = GetAxis(prev_trans_op);
      IR_ENFORCE(axis_first.size() == axis_last.size(),
                 "tranpose op's perm rank should be same.");
      auto new_perm = GetPerm(axis_first, axis_last);
      rewriter.SetInsertionPoint(op);
      auto new_transpose_op = rewriter.Build<paddle::dialect::TransposeOp>(
          ir::GetDefiningOpForInput<0>(prev_trans_op)->result(0), new_perm);
      rewriter.ReplaceOp(op, {new_transpose_op.out()});
      return true;
    }

    return false;
  }

 private:
  std::vector<int> GetAxis(paddle::dialect::TransposeOp op) const {
    auto array_attr = op.attribute<ir::ArrayAttribute>("perm").data();
    std::vector<int> axis(array_attr.size());
    for (size_t i = 0; i < array_attr.size(); ++i) {
      axis[i] = array_attr[i].dyn_cast<ir::Int32Attribute>().data();
    }
    return axis;
  }

  std::vector<int> GetPerm(const std::vector<int> &perm1,
                           const std::vector<int> &perm2) const {
    int n = perm1.size();
    std::vector<int> axis(n), axis1(n), axis2(n);
    std::iota(axis.begin(), axis.end(), 0);
    for (int i = 0; i < n; ++i) {
      axis1[i] = axis[perm1[i]];
    }
    for (int i = 0; i < n; ++i) {
      axis2[i] = axis1[perm2[i]];
    }
    return axis2;
  }
};

class Conv2dBnFusePattern
    : public ir::OpRewritePattern<paddle::dialect::BatchNormOp> {
 public:
  using ir::OpRewritePattern<paddle::dialect::BatchNormOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::dialect::BatchNormOp op,
      ir::PatternRewriter &rewriter) const override {  // NOLINT
    // The next op should be batch_norm.
    paddle::dialect::Conv2dOp conv2d_op =
        ir::GetDefiningOpForInput(op)->dyn_cast<paddle::dialect::Conv2dOp>();
    if (!conv2d_op) return false;

    ir::OpResult conv2d_out = conv2d_op.out();
    if (!conv2d_out.HasOneUse()) return false;

    ir::Value conv2d_filter = conv2d_op.filter();

    // ir::GetParameterOp filter_parameter_op =
    //     conv2d_filter.GetDefiningOp()->dyn_cast<ir::GetParameterOp>();
    // if (!filter_parameter_op) return false;

    ir::OpResult conv2d_filter_result = conv2d_filter.dyn_cast<ir::OpResult>();
    IR_ENFORCE(conv2d_filter_result);

    ir::Value bn_input = op.x();
    IR_ENFORCE(bn_input == conv2d_out);

    ir::Value bn_mean = op.mean();
    ir::Value bn_variance = op.variance();
    ir::Value bn_scale = op.scale();
    ir::Value bn_bias = op.bias();

    ir::OpResult bn_mean_result = bn_mean.dyn_cast<ir::OpResult>();
    IR_ENFORCE(bn_mean_result);
    ir::OpResult bn_variance_result = bn_variance.dyn_cast<ir::OpResult>();
    IR_ENFORCE(bn_variance_result);
    ir::OpResult bn_scale_result = bn_scale.dyn_cast<ir::OpResult>();
    IR_ENFORCE(bn_scale_result);
    ir::OpResult bn_bias_result = bn_bias.dyn_cast<ir::OpResult>();
    IR_ENFORCE(bn_bias_result);

    // --- deal with filter ---
    rewriter.SetInsertionPoint(conv2d_op);
    phi::DDim bn_variance_shape =
        bn_variance.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    float epsilon = op.attribute<ir::FloatAttribute>("epsilon").data();
    paddle::dialect::FullOp full_op = rewriter.Build<paddle::dialect::FullOp>(
        phi::vectorize(bn_variance_shape), epsilon);
    paddle::dialect::AddOp add_op = rewriter.Build<paddle::dialect::AddOp>(
        bn_variance_result, full_op.out());
    paddle::dialect::SqrtOp sqrt_op =
        rewriter.Build<paddle::dialect::SqrtOp>(add_op.out());
    paddle::dialect::DivideOp div_op =
        rewriter.Build<paddle::dialect::DivideOp>(bn_scale_result,
                                                  sqrt_op.out());

    // reshape scale
    phi::DDim conv2d_filter_shape = ir::GetShapeFromValue(conv2d_filter);
    phi::DDim bn_scale_shape =
        bn_scale.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    std::vector<int64_t> bn_scale_new_shape(conv2d_filter_shape.size(), 1);
    bn_scale_new_shape[0] = bn_scale_shape[0];

    paddle::dialect::ReshapeOp reshape_scale_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(div_op.out(),
                                                   bn_scale_new_shape);
    // new filter --> mul_op.out()
    paddle::dialect::MultiplyOp mul_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(conv2d_filter_result,
                                                    reshape_scale_op.out());
    // TODO(liuyuanle): Use rewriter.
    conv2d_op->op_operand(1).set_source(mul_op.out());

    // --- deal with bias ---
    rewriter.SetInsertionPoint(op);
    paddle::dialect::MultiplyOp mul_bias_op =
        rewriter.Build<paddle::dialect::MultiplyOp>(bn_mean_result,
                                                    div_op.out());
    // new bias --> sub_op.out()
    paddle::dialect::SubtractOp sub_op =
        rewriter.Build<paddle::dialect::SubtractOp>(bn_bias_result,
                                                    mul_bias_op.out());

    // reshape new bias
    phi::DDim conv2d_out_shape = ir::GetShapeFromValue(conv2d_out);
    std::vector<int64_t> new_bias_new_shape(conv2d_out_shape.size(), 1);
    std::string data_format =
        conv2d_op.attribute<ir::StrAttribute>("data_format").data();

    IR_ENFORCE(data_format == "NCHW", "Only support NCHW now.");
    new_bias_new_shape[0] = conv2d_out_shape[0];
    new_bias_new_shape[1] = conv2d_out_shape[1];

    paddle::dialect::ReshapeOp reshape_bias_op =
        rewriter.Build<paddle::dialect::ReshapeOp>(sub_op.out(),
                                                   new_bias_new_shape);

    paddle::dialect::AddOp add_bias_op = rewriter.Build<paddle::dialect::AddOp>(
        conv2d_out, reshape_bias_op.out());
    auto next_op = ir::GetFirstUseOperationForOutput<0>(op);
    rewriter.ReplaceAllUsesWith(next_op->operand(0), add_bias_op.out());

    rewriter.EraseOp(op);
    return true;
  }
};

class TestPass : public ir::Pass {
 public:
  TestPass() : ir::Pass("TestPass", 1) {}
  void Run(ir::Operation *op) override {
    ir::RewritePatternSet ps(op->ir_context());
    ps.Add<TransposePatternRewrite>(op->ir_context());
    ps.Add<Conv2dBnFusePattern>(op->ir_context());

    ir::FrozenRewritePatternSet frozen_ps(std::move(ps));
    ir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    ir::ApplyPatternsGreedily(op->region(0), frozen_ps, cfg);
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

void BuildProgram(ir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 3, 16, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_filter_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 3, 3, 3},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_mean_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{64}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp full_variance_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_scale_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_bias_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{64}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::Conv2dOp conv2d_op =
      builder.Build<paddle::dialect::Conv2dOp>(full_input_op.out(),
                                               full_filter_op.out());

  paddle::dialect::BatchNormOp batch_norm_op =
      builder.Build<paddle::dialect::BatchNormOp>(conv2d_op.out(),
                                                  full_mean_op.out(),
                                                  full_variance_op.out(),
                                                  full_scale_op.out(),
                                                  full_bias_op.out(),
                                                  true,
                                                  0.9,
                                                  1e-6,
                                                  "NCHW",
                                                  false,
                                                  false);

  auto transpose1_op = builder.Build<paddle::dialect::TransposeOp>(
      batch_norm_op.out(), std::vector<int>{0, 2, 3, 1});

  auto transpose2_op = builder.Build<paddle::dialect::TransposeOp>(
      transpose1_op.out(), std::vector<int>{0, 3, 1, 2});

  builder.Build<paddle::dialect::FetchOp>(transpose2_op.out(), "out");
}

// TODO(wilber): Add a normal test.
TEST(pattern_rewrite, Patterns) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder = ir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 11u);

  ir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<TestPass>());
  pm.AddPass(ir::CreateDCEPass());
  program.Print(std::cout);
  std::cout << std::endl;
  pm.Run(&program);
  LOG(INFO) << "After Pass.";
  program.Print(std::cout);
  std::cout << std::endl;
}
