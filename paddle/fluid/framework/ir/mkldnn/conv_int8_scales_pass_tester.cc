// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_int8_scales_pass.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);
  if (type == "conv2d") {
    const std::vector<int> strides({1, 1});
    const std::vector<int> paddings({0, 0});
    const std::vector<int> dilations({1, 1});
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("name", name);
    op->SetAttr("strides", strides);
    op->SetAttr("groups", 1);
    op->SetAttr("paddings", paddings);
    op->SetAttr("padding_algorithm", std::string("EXPLICIT"));
    op->SetAttr("dilations", dilations);
    op->SetAttr("data_format", std::string("NCHW"));

    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2)
      op->SetInput("Bias", {inputs[2]});
    else
      op->SetInput("Bias", {});

    op->SetOutput("Output", outputs);
    op->SetAttr("Scale_in", 1.0f);
    op->SetAttr("Scale_out", 1.0f);
    op->SetAttr("Scale_weights", std::vector<float>{1.5f});
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("mkldnn_data_type", std::string("int8"));
  } else {
    FAIL() << "Unexpected operator type.";
  }
}

ProgramDesc BuildProgramDesc(bool convWithExistingBias) {
  ProgramDesc prog;
  std::vector<std::string> nodes{"c", "weights", "f"};
  if (convWithExistingBias) nodes.push_back("conv_bias");
  for (auto& v : nodes) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
    if (v == "weights") {
      var->SetPersistable(true);
    }
  }

  if (convWithExistingBias) {
    SetOp(&prog, "conv2d", "conv",
          std::vector<std::string>({"c", "weights", "conv_bias"}),
          std::vector<std::string>({"f"}));
  } else {
    SetOp(&prog, "conv2d", "conv", std::vector<std::string>({"c", "weights"}),
          std::vector<std::string>({"f"}));
  }

  return prog;
}

void MainTest(bool convWithExistingBias, int removed_nodes_count, float scale,
              float scale_weight) {
  auto prog = BuildProgramDesc(convWithExistingBias);
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("conv_int8_scales_pass");
  int original_nodes_num = graph->Nodes().size();
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_EQ(original_nodes_num, current_nodes_num);

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));

      EXPECT_EQ(op->GetAttrIfExists<std::vector<float>>("Scale_weights")[0],
                scale_weight);
      EXPECT_EQ(op->GetAttrIfExists<float>("Scale_in"), scale);
      EXPECT_EQ(op->GetAttrIfExists<float>("Scale_out"), scale);

      EXPECT_EQ(op->GetAttrIfExists<float>("Sum_scale"), scale);
      EXPECT_EQ(
          op->GetAttrIfExists<std::vector<float>>("Output_shift_scale")[0],
          scale / scale_weight);
      EXPECT_EQ(op->GetAttrIfExists<float>("Activation_scale"), scale);

      if (convWithExistingBias) {
        EXPECT_EQ(op->GetAttrIfExists<std::vector<float>>("Bias_scales")[0],
                  scale * scale_weight);
      }
    }
  }
  EXPECT_EQ(original_nodes_num - removed_nodes_count, current_nodes_num);
}

TEST(ConvInt8ScalesPass, conv_int8_with_no_bias) {
  auto scale = 1.0f;
  int removed_nodes_count = 0;
  auto scale_weight = 1.5;
  MainTest(false, removed_nodes_count, scale, scale_weight);
}

TEST(ConvInt8ScalesPass, conv_int8_with_bias) {
  auto scale = 1.0f;
  int removed_nodes_count = 0;
  auto scale_weight = 1.5f;
  MainTest(true, removed_nodes_count, scale, scale_weight);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_int8_scales_pass);
