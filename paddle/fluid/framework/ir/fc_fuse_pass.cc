// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fc_fuse_pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FCFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);
  FusePassBase::Init("fc_fuse", graph);
  int found_fc_count = 0;

  for (bool with_relu : {true, false}) {
    GraphPatternDetector gpd;
    auto* x = gpd.mutable_pattern()
                  ->NewNode("fc_fuse/x")
                  ->AsInput()
                  ->assert_is_op_input("mul", "X");
    patterns::FC fc_pattern(gpd.mutable_pattern(), "fc_fuse");
    fc_pattern(x, true /*with bias*/, with_relu);

    auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                       Graph* g) {
      if (subgraph.count(x) <= 0) {
        LOG(WARNING) << "The subgraph is empty.";
        return;
      }

      VLOG(4) << "handle FC fuse";
      GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                                fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, fc_pattern);
      Node* relu = nullptr;
      Node* relu_out = nullptr;
      if (with_relu) {
        GET_IR_NODE_FROM_SUBGRAPH(tmp_relu, relu, fc_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(tmp_relu_out, relu_out, fc_pattern);
        relu = tmp_relu;
        relu_out = tmp_relu_out;
      }

      auto base_op_desc = mul->Op();
      // Create an FC Node.
      // OpDesc desc(base_op_desc, nullptr);
      OpDesc desc;
      std::string fc_x_name = subgraph.at(x)->Name();
      std::string fc_w_name = w->Name();
      std::string fc_bias_name = bias->Name();
      std::string fc_out_name =
          with_relu ? relu_out->Name() : elementwise_add_out->Name();

      desc.SetInput("Input", std::vector<std::string>({fc_x_name}));
      desc.SetInput("W", std::vector<std::string>({fc_w_name}));
      desc.SetInput("Bias", std::vector<std::string>({fc_bias_name}));
      desc.SetOutput("Out", std::vector<std::string>({fc_out_name}));
      desc.SetAttr("in_num_col_dims", mul->Op()->GetAttr("x_num_col_dims"));
      if (with_relu) {
        std::string activation_type = "relu";
        desc.SetAttr("activation_type", activation_type);
      } else {
        std::string activation_type = "";
        desc.SetAttr("activation_type", activation_type);
      }

      // For anakin subgraph int8
      // When in anakin subgraph int8 mode, the pattern like "fake_quant + mul +
      // fake_dequant"
      // can be detected by the quant_dequant_fuse_pass. This pass will add
      // "input_scale",
      // "weight_scale" which are extracted from fake_quant op and fake_dequant
      // op
      // to mul op,
      // and then delete the fake_quant op and fake_dequant op in the graph. If
      // the mul op
      // has the scale info, we should add those to the fused fc.
      if (base_op_desc->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", base_op_desc->GetAttr("enable_int8"));
        desc.SetAttr("input_scale", base_op_desc->GetAttr("input_scale"));
        desc.SetAttr("weight_scale", base_op_desc->GetAttr("weight_scale"));
        if (base_op_desc->HasAttr("out_scale"))
          desc.SetAttr("out_scale", base_op_desc->GetAttr("out_scale"));
        auto elementwise_desc = elementwise_add->Op();
        if (elementwise_desc->HasAttr("out_scale"))
          desc.SetAttr("out_scale", elementwise_desc->GetAttr("out_scale"));
      }

      desc.SetType("fc");

      auto fc_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
      if (with_relu) {
        GraphSafeRemoveNodes(
            graph, {mul, elementwise_add, mul_out, elementwise_add_out, relu});
      } else {
        GraphSafeRemoveNodes(graph, {mul, elementwise_add, mul_out});
      }

      IR_NODE_LINK_TO(subgraph.at(x), fc_node);
      IR_NODE_LINK_TO(w, fc_node);
      IR_NODE_LINK_TO(bias, fc_node);
      if (with_relu) {
        IR_NODE_LINK_TO(fc_node, relu_out);
      } else {
        IR_NODE_LINK_TO(fc_node, elementwise_add_out);
      }

      found_fc_count++;
    };

    gpd(graph, handler);
    LOG(INFO) << "with_relu: " << with_relu
              << ", found_fc_count: " << found_fc_count;
  }

  AddStatis(found_fc_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_fuse_pass, paddle::framework::ir::FCFusePass);
