/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(range);
USE_OP_DEVICE_KERNEL(range, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type) {
  // init
  auto start = scope->Var("start");
  auto tensor_start = start->GetMutable<f::LoDTensor>();

  auto end = scope->Var("end");
  auto tensor_end = end->GetMutable<f::LoDTensor>();

  auto step = scope->Var("step");
  auto tensor_step = step->GetMutable<f::LoDTensor>();

  std::vector<T> init_start;
  init_start.push_back(static_cast<T>(1));

  std::vector<T> init_end;
  init_end.push_back(static_cast<T>(10));

  std::vector<T> init_step;
  init_step.push_back(static_cast<T>(2));

  TensorFromVector(init_start, ctx, tensor_start);
  TensorFromVector(init_end, ctx, tensor_end);
  TensorFromVector(init_step, ctx, tensor_step);

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(op_type, {{"start", {"start"}}, {"end", {"end"}}},
                                    {{"step", {"step"}}}, attrs);

  op->Run(*scope, place);

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  std::vector<T> expected_vec;
  expected_vec.push_back(static_cast<T>(1.0));
  expected_vec.push_back(static_cast<T>(3.0));
  expected_vec.push_back(static_cast<T>(5.0));
  expected_vec.push_back(static_cast<T>(7.0));
  expected_vec.push_back(static_cast<T>(9.0));


  EXPECT_EQ(expected_vec.size(), out_vec.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], static_cast<T>(expected_vec[i]));
  }
}


TEST(range, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<int>(&scope, ctx, "range");
}


