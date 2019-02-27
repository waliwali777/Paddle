/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <map>

#include "framework/core/net/net.h"
#include "framework/graph/graph.h"
#include "framework/graph/graph_global_mem.h"
#include "paddle/fluid/inference/anakin/engine.h"

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::Precision;
using anakin::saber::NV;
using anakin::saber::X86;
using anakin::saber::Shape;
using anakin::PBlock;
using anakin::PTuple;
namespace paddle {
namespace inference {
namespace anakin {

class TestAnakinEngine : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {}

 protected:
  using AnakinNvEngineT = AnakinEngine<NV, Precision::FP32>;
  std::unique_ptr<AnakinNvEngineT> engine_{nullptr};
};

void TestAnakinEngine::SetUp() {
  engine_.reset(new AnakinEngine<NV, Precision::FP32>(true));
}

TEST_F(TestAnakinEngine, Execute) {
  engine_->AddOp("op1", "Dense", {"x"}, {"y"});
  engine_->AddOpAttr("op1", "out_dim", 2);
  engine_->AddOpAttr("op1", "bias_term", false);
  engine_->AddOpAttr("op1", "axis", 1);
  std::vector<int> shape = {1, 1, 1, 2};
  Shape tmp_shape(shape);
  // PBlock<NV> weight1(tmp_shape);
  auto *weight1 =
      GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(tmp_shape);
  // auto *weight1 = new PBlock<NV>(tmp_shape, AK_FLOAT);

  float *cpu_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  cpu_data[0] = 2.;
  weight1->d_tensor().set_shape(tmp_shape);
  weight1->d_tensor().copy_from(weight1->h_tensor());
  engine_->AddOpAttr("op1", "weight_1", *weight1);

  engine_->Freeze();
  // PTuple<int> input_shape = {1};
  // engine_->AddOpAttr("x", "input_shape", input_shape);
  engine_->SetInputShape("x", {1, 1, 1, 1});
  engine_->Optimize();
  engine_->InitGraph();
  framework::LoDTensor x;
  framework::LoDTensor y;
  x.Resize({1, 1, 1, 1});
  y.Resize({1, 1, 1, 2});
  auto *x_data = x.mutable_data<float>(platform::CUDAPlace());
  float x_data_cpu[] = {1.};
  cudaMemcpy(x_data, x_data_cpu, sizeof(float), cudaMemcpyHostToDevice);

  std::map<std::string, framework::LoDTensor *> inputs = {{"x", &x}};
  auto *y_data = y.mutable_data<float>(platform::CUDAPlace());
  std::map<std::string, framework::LoDTensor *> outputs = {{"y", &y}};

  engine_->Execute(inputs, outputs);
  auto *y_data_gpu = y_data;
  float y_data_cpu[2];
  cudaMemcpy(y_data_cpu, y_data_gpu, sizeof(float) * 2, cudaMemcpyDeviceToHost);
  LOG(INFO) << "output value: " << y_data_cpu[0] << ", " << y_data_cpu[1];
}

/*
TEST_F(TestAnakinEngine, Execute1) {
    std::unique_ptr<AnakinNvEngineT> engine(new AnakinEngine<NV,
Precision::FP32>(true));
    engine->AddOp("op1", "Dense", {"x"}, {"y"});
    engine->AddOpAttr("op1", "out_dim", 2);
    engine->AddOpAttr("op1", "bias_term", false);
    // axis M * K
    engine->AddOpAttr("op1", "axis", 3);
    std::vector<int> shape = {1, 1, 3, 2};
    ::anakin::saber::Shape tmp_shape(shape);

    ::anakin::PBlock<::anakin::saber::NV> weight1(tmp_shape);
    float *cpu_data = static_cast<float *>(weight1.h_tensor().mutable_data());
    for (int i = 0; i < 2 * 3; i++) {
        cpu_data[i] = i + 1;
    }
    weight1.d_tensor().set_shape(tmp_shape);
    weight1.d_tensor().copy_from(weight1.h_tensor());
    engine->AddOpAttr("op1", "weight_1", weight1);

    ::anakin::PTuple<int> input_shape = {1, 1, 1, 3};
  //PTuple<int> input_shape = {1};
  //engine->AddOpAttr("x", "input_shape", input_shape);
  engine->Freeze();
  engine->SetInputShape("x", {1, 1, 1, 3});
  engine->Optimize();
  engine->InitGraph();
  framework::LoDTensor x;
  framework::LoDTensor y;
  x.Resize({1, 1, 1, 1});
  y.Resize({1, 1, 1, 2});
  auto *x_data = x.mutable_data<float>(platform::CUDAPlace());
  float x_data_cpu[] = {1.};
  cudaMemcpy(x_data, x_data_cpu, sizeof(float), cudaMemcpyHostToDevice);

  std::map<std::string, framework::LoDTensor *> inputs = {{"x", &x}};
  auto *y_data = y.mutable_data<float>(platform::CUDAPlace());
  if (y_data == nullptr) {}
  std::map<std::string, framework::LoDTensor *> outputs = {{"y", &y}};
  //ASSERT_EQ(inputs.size(), 4);
  //ASSERT_EQ(outputs.size(), 4);
  engine->Execute(inputs, outputs);
  auto *y_data_gpu = y.data<float>();
  float y_data_cpu;
  cudaMemcpy(&y_data_cpu, y_data_gpu, sizeof(float), cudaMemcpyDeviceToHost);
  LOG(INFO) << "output value: " << y_data_cpu;
}
*/

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
