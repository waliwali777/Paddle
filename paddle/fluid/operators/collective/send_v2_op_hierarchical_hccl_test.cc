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

#include <stdio.h>
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

#include "paddle/fluid/operators/collective/c_hierarchical_hccl_collective_test_help.h"
#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_HIERARCHICAL_HCCL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(send_v2);
USE_OP_DEVICE_KERNEL(send_v2, NPU);

void TestHcomSendOp(f::Scope* scope, const p::DeviceContext& ctx) {
  std::cout << "BEGIN TEST:" << __FUNCTION__ << std::endl;
  auto x = scope->Var("Data");
  auto tensor_x = x->GetMutable<f::LoDTensor>();
  int num = FLAGS_data_size;

  EXPECT_GT(num, 0);
  EXPECT_LT(num, 1 << 15);
  std::vector<float> init(num * num, 1.0 * FLAGS_dest_rank);
  int rank_id = FLAGS_rank_id;
  VLOG(3) << "rank id:" << rank_id;
  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num, num});
  ctx.Wait();
  auto place = ctx.GetPlace();
  ctx.Wait();

  f::AttributeMap attrs;
  attrs["tag"] = std::string("srtest");
  attrs["peer"] = FLAGS_dest_rank;
  attrs["ring_id"] = 0;
  attrs["srTag"] = 0;
  attrs["use_calc_stream"] = 1;

  auto op = f::OpRegistry::CreateOp("send_v2", {{"X", {"Data"}}}, {}, attrs);

  for (int i = 0; i < 10; i++) {
    op->Run(*scope, place);
  }
  VLOG(3) << "send run over";
  ctx.Wait();
}

TEST(send_v2, NPU) {
  check_test_sendrecv_env();
  f::Scope scope;
  HierarchicalHcclCommGroupIdType group_name = "test_group";

  char* npu_id = getenv("FLAGS_selected_npus");
  VLOG(3) << "Select npu:" << npu_id;
  p::NPUDeviceContext ctx(p::NPUPlace(atoi(npu_id)));

  prepare(&scope, ctx, group_name);
  TestHcomSendOp(&scope, ctx);
}
