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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/utils/data_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"

using cinn::hlir::framework::pir::Group;
using cinn::hlir::framework::pir::GroupPtr;

using ProgramInfo =
    std::tuple<std::shared_ptr<::pir::Program>, std::vector<GroupPtr>>;
ProgramInfo BuildProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value_one = 1.0;  // relu(tan(1.)) = 1.5;
  const float value_two = 2.0;  // relu(tan(2.)) = 0.
  auto full_op_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             value_one,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto full_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             value_two,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(full_op_x->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));

  std::vector<GroupPtr> groups;
  groups.emplace_back(
      std::make_shared<Group>(std::initializer_list<::pir::Operation*>(
          {full_op_x.operation()})));  // For coverage
  groups.emplace_back(std::make_shared<Group>(
      std::initializer_list<::pir::Operation*>({full_op_y.operation()})));
  groups.emplace_back(std::make_shared<Group>(
      std::vector<::pir::Operation*>({tan_op_x.operation(),
                                      relu_op_x.operation(),
                                      tan_op_y.operation(),
                                      relu_op_y.operation()})));

  return {program, groups};
}

ProgramInfo BuildSoftmax() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  std::vector<int64_t> axes{-1};

  auto x =
      builder
          .Build<paddle::dialect::FullOp>(std::vector<int64_t>({128, 128, 768}),
                                          1.0,
                                          phi::DataType::FLOAT32,
                                          phi::GPUPlace(0))
          .result(0);
  auto max = builder.Build<cinn::dialect::ReduceMaxOp>(x, axes, true).result(0);
  auto broadcast_1 = builder
                         .Build<cinn::dialect::BroadcastOp>(
                             max,
                             std::vector<int64_t>({0, 1, 2}),
                             std::vector<int64_t>({128, 128, 768}))
                         .result(0);
  auto sub =
      builder.Build<paddle::dialect::SubtractOp>(x, broadcast_1).result(0);
  auto exp = builder.Build<paddle::dialect::ExpOp>(sub).result(0);
  auto sum =
      builder.Build<cinn::dialect::ReduceSumOp>(exp, axes, true).result(0);

  auto broadcast_2 = builder
                         .Build<cinn::dialect::BroadcastOp>(
                             sum,
                             std::vector<int64_t>({0, 1, 2}),
                             std::vector<int64_t>({128, 128, 768}))
                         .result(0);
  auto divide =
      builder.Build<paddle::dialect::DivideOp>(exp, broadcast_2).result(0);

  std::vector<GroupPtr> groups;
  groups.emplace_back(std::make_shared<Group>(
      std::initializer_list<::pir::Operation*>({x.owner()})));
  groups.emplace_back(std::make_shared<Group>(
      std::initializer_list<::pir::Operation*>({max.owner(),
                                                broadcast_1.owner(),
                                                sub.owner(),
                                                exp.owner(),
                                                sum.owner(),
                                                broadcast_2.owner(),
                                                divide.owner()})));

  return {program, groups};
}

TEST(PIRCompier, CompileSoftmax) {
  // Step 1: Construct pir::Program
  auto prog_info = BuildSoftmax();
  std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
  std::vector<GroupPtr> groups = std::get<1>(prog_info);
  EXPECT_EQ(program->block()->size(), 8u);
  LOG(INFO) << program->block()->size();

  std::stringstream ss;
  program->Print(ss);
  LOG(INFO) << ss.str();

  // Step 2: Compiler New pir::Program into Runtime Program
  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);
  LOG(INFO) << scope->var_names().size();
  ASSERT_EQ(scope->var_names().size(), 8);

  cinn::hlir::framework::PIRCompiler ir_compiler(*program, target, scope);
  auto runtime_program = ir_compiler.Build(groups);

  // Step 3: Execute Runtime Instruction and check Scope.
  ASSERT_NO_THROW(runtime_program->Execute());
  for (auto& var_name : scope->var_names()) {
    std::string name = {var_name.begin(), var_name.end()};
    std::vector<float> data =
        cinn::GetTensorData<float>(scope->GetTensor(name), target);
    for (int i = 0; i < 1; ++i) {
      LOG_FIRST_N(INFO, 10) << "data: " << data[i];
    }
  }
}

// TEST(PIRCompier, CompilerAndRun) {
//   // Step 1: Construct pir::Program
//   auto prog_info = BuildProgram();
//   std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
//   EXPECT_EQ(program->block()->size(), 6u);
//   LOG(INFO) << program->block()->size();

//   std::stringstream ss;
//   program->Print(ss);
//   LOG(INFO) << ss.str();

//   // Step 2: Compiler New pir::Program into Runtime Program
//   auto target = cinn::common::DefaultNVGPUTarget();
//   auto scope = cinn::hlir::framework::BuildScope(target, *program);
//   ASSERT_EQ(scope->var_names().size(), 6);

//   cinn::hlir::framework::PIRCompiler ir_compiler(*program, target, scope);
//   auto runtime_program = ir_compiler.Build();

//   // Step 3: Execute Runtime Instruction and check Scope.
//   ASSERT_NO_THROW(runtime_program->Execute());
//   for (auto& var_name : scope->var_names()) {
//     std::string name = {var_name.begin(), var_name.end()};
//     std::vector<float> data =
//         cinn::GetTensorData<float>(scope->GetTensor(name), target);
//     for (int i = 0; i < 1; ++i) {
//       LOG_FIRST_N(INFO, 10) << "data: " << data[i];
//     }
//   }
// }

// TEST(PIRCompier, CompileGroupOps) {
//   // Step 1: Construct pir::Program
//   auto prog_info = BuildProgram();
//   std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
//   std::vector<GroupPtr> groups = std::get<1>(prog_info);
//   EXPECT_EQ(program->block()->size(), 6u);
//   LOG(INFO) << program->block()->size();

//   std::stringstream ss;
//   program->Print(ss);
//   LOG(INFO) << ss.str();

//   // Step 2: Compiler New pir::Program into Runtime Program
//   auto target = cinn::common::DefaultNVGPUTarget();
//   auto scope = cinn::hlir::framework::BuildScope(target, *program);
//   ASSERT_EQ(scope->var_names().size(), 6);

//   cinn::hlir::framework::PIRCompiler ir_compiler(*program, target, scope);
//   auto runtime_program = ir_compiler.Build(groups);

//   // Step 3: Execute Runtime Instruction and check Scope.
//   ASSERT_NO_THROW(runtime_program->Execute());
//   for (auto& var_name : scope->var_names()) {
//     std::string name = {var_name.begin(), var_name.end()};
//     std::vector<float> data =
//         cinn::GetTensorData<float>(scope->GetTensor(name), target);
//     for (int i = 0; i < 1; ++i) {
//       LOG_FIRST_N(INFO, 10) << "data: " << data[i];
//     }
//   }
// }

// TEST(RuntimeDialect, CompilerAndRun) {
//   // Step 1: Construct pir::Program
//   auto prog_info = BuildProgram();
//   std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
//   EXPECT_EQ(program->block()->size(), 6u);

//   // Step 2: Compiler New pir::Program into Runtime Program
//   auto target = cinn::common::DefaultNVGPUTarget();
//   auto scope = cinn::hlir::framework::BuildScope(target, *program);
//   ASSERT_EQ(scope->var_names().size(), 6u);

//   cinn::hlir::framework::PIRCompiler ir_compiler(*program, target, scope);
//   auto runtime_program = ir_compiler.Build();
// }
