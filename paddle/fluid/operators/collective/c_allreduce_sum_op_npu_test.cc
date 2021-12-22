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

#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#endif

// Node1: HCCL_WHITELIST_DISABLE=1 FLAGS_selected_npus=1 GLOG_v=4 RANK_ID=1
// DEVICE_ID=1 ./paddle/fluid/operators/collective/c_allreduce_sum_op_npu_test
// Node2: HCCL_WHITELIST_DISABLE=1 FLAGS_selected_npus=0 GLOG_v=4 RANK_ID=0
// DEVICE_ID=0 ./paddle/fluid/operators/collective/c_allreduce_sum_op_npu_test

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(c_allreduce_sum);
USE_NO_KERNEL_OP(c_gen_hccl_id);
USE_NO_KERNEL_OP(c_comm_init_hccl);
USE_OP_DEVICE_KERNEL(c_allreduce_sum, NPU);

DECLARE_string(selected_npus);

template <typename T>
void PrintDebugInfo(const std::string preStr, const std::vector<T>& data) {
  std::string debugstring = "";
  std::cout << preStr << ":" << std::endl << debugstring;
  for (auto ele : data) {
    std::cout << ele << " ";
  }
  std::cout << std::endl;
}

void PrepareUniqueId(f::Scope* scope, const p::DeviceContext& ctx,
                     HcclRootInfo* hccl_id) {
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_id = " << rank_id
          << "; RANK_TABLE_FILE = " << atoi(getenv("DEVICE_ID"));

  std::vector<int> rank_ids{0, 1};
  f::AttributeMap gen_hccl_id;

  std::vector<std::string> endpointList = {"127.0.0.1:6175", "127.0.0.1:6177"};
  gen_hccl_id["rank"] = rank_id;
  gen_hccl_id["endpoint"] = endpointList[rank_id];
  std::vector<std::string> other_endpoints = {
      endpointList[rank_id == 0 ? 1 : 0]};
  gen_hccl_id["other_endpoints"] = other_endpoints;

  auto out = scope->Var("Out");
  auto id = out->GetMutable<HcclRootInfo>();

  VLOG(3) << "break";

  auto comm_init_op = f::OpRegistry::CreateOp("c_gen_hccl_id", {},
                                              {{"Out", {"Out"}}}, gen_hccl_id);
  VLOG(3) << "break";
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();

  memcpy(hccl_id, id, 1024);
}

void Prepare(f::Scope* scope, const p::DeviceContext& ctx,
             HcclRootInfo* hccl_id) {
  auto x = scope->Var("X");
  auto id = x->GetMutable<HcclRootInfo>();

  memcpy(id, hccl_id, 1024);

  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_id = " << rank_id
          << "; RANK_TABLE_FILE = " << atoi(getenv("DEVICE_ID"));

  // std::vector<int> rank_ids{0, 1};
  f::AttributeMap comm_init_attrs;
  comm_init_attrs["ring_id"] = 0;
  comm_init_attrs["rank_ids"] = 2;
  comm_init_attrs["rank"] = rank_id;
  comm_init_attrs["device_id"] = device_id;
  // comm_init_attrs["rank_ids"] = rank_ids;
  auto comm_init_op = f::OpRegistry::CreateOp(
      "c_comm_init_hccl", {{"X", {"X"}}}, {}, comm_init_attrs);
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();
}

template <typename T>
void TestHCCLAllReduceOp(f::Scope* scope, const p::DeviceContext& ctx,
                         int iter) {
  // init
  auto x = scope->Var("Data");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  int rank_id = atoi(getenv("RANK_ID"));
  int num1 = 3;
  int num2 = 128;

  std::vector<T> init;
  for (int64_t i = 0; i < num1 * num2; ++i) {
    init.push_back(static_cast<T>(1.0 + rank_id));
  }
  init[0] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
  PrintDebugInfo("input data", init);

  auto place = ctx.GetPlace();

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num1, num2});
  ctx.Wait();

  auto out = scope->Var("OutData");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num1, num2});
  tensor_out->mutable_data<T>(place);  // allocate
  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs["tag"] = std::string("tagx_" + std::to_string(iter));
  attrs["ring_id"] = 0;
  attrs["use_calc_stream"] = 1;

  auto op = f::OpRegistry::CreateOp("c_allreduce_sum", {{"X", {"Data"}}},
                                    {{"Out", {"OutData"}}}, attrs);
  for (int i = 0; i < 1; i++) {
    op->Run(*scope, place);
  }
  ctx.Wait();

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);
  ctx.Wait();

  PrintDebugInfo("output data", out_vec);

  float diff = static_cast<float>(out_vec[0]) - 65504;
  EXPECT_TRUE(diff < 0.1 && diff > -0.1);
  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 1; i < 10; i++) {
    EXPECT_EQ(out_vec[i], static_cast<paddle::platform::float16>(3.0));
  }
}

TEST(c_allreduce_sum, NPU) {
  f::Scope scope;
  HcclRootInfo hccl_id;

  p::NPUDeviceContext ctx(p::NPUPlace(atoi(FLAGS_selected_npus.c_str())));

  // only support one device, if more than one device, use first default
  PrepareUniqueId(&scope, ctx, &hccl_id);
  Prepare(&scope, ctx, &hccl_id);

  TestHCCLAllReduceOp<paddle::platform::float16>(&scope, ctx, 1);
  // TestHCCLAllReduceOp<float>(&scope, ctx, 0);
}
