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

#include "paddle/fluid/framework/ir/graph.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class NOP : public OperatorBase {
 public:
  NOP(const std::string &type, const VariableNameMap &inputs,
      const VariableNameMap &outputs, const AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope &scope,
               const platform::Place &place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(const OpDesc &op_desc, BlockDesc *block) const override {
    auto &inputs = op_desc.Input("X");
    auto default_var_type = proto::VarType::SELECTED_ROWS;

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [block](const std::string &name) {
          return block->Var(name)->GetType() == proto::VarType::LOD_TENSOR;
        });
    if (any_input_is_lod_tensor) {
      default_var_type = proto::VarType::LOD_TENSOR;
    }

    auto out_var_name = op_desc.Output("Out").front();
    block->Var(out_var_name)->SetType(default_var_type);
  }
};
}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(sum, paddle::framework::NOP, paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(sum_without_infer_var_type, paddle::framework::NOP,
                  paddle::framework::SumOpMaker);

namespace paddle {
namespace framework {

TEST(GraphTest, Basic) {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("test_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(proto::VarType::SELECTED_ROWS,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::LOD_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(proto::VarType::LOD_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  std::vector<ir::Node *> nodes(g->Nodes().begin(), g->Nodes().end());
  for (ir::Node *n : nodes) {
    if (n->Name() == "sum") {
      ASSERT_EQ(n->inputs.size(), 3);
      ASSERT_EQ(n->outputs.size(), 1);
    } else if (n->Name() == "test_a" || n->Name() == "test_b" ||
               n->Name() == "test_c") {
      ASSERT_EQ(n->inputs.size(), 0);
      ASSERT_EQ(n->outputs.size(), 1);
    } else if (n->Name() == "test_out") {
      ASSERT_EQ(n->inputs.size(), 1);
      ASSERT_EQ(n->outputs.size(), 0);
    }
  }
  ASSERT_EQ(nodes.size(), 5);
}
}  // namespace framework
}  // namespace paddle
