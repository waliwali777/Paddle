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

#pragma once

#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Convert Op from Fluid to TensorRT Engine.
 */
class OpConverter {
 public:
  OpConverter() {}

  // Converter logic for an op.
  virtual void operator()(const framework::proto::OpDesc& op,
                          const framework::Scope& scope) {}

  // Convert a single fluid operaotr and add the corresponding layer to TRT.
  void ConvertOp(const framework::proto::OpDesc& op,
                 const std::unordered_set<std::string>& parameters,
                 const framework::Scope& scope, TensorRTEngine* engine) {
    framework::OpDesc op_desc(op, nullptr, nullptr);

    OpConverter* it{nullptr};

    if (op_desc.Type() == "mul") {
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1UL);
      std::string Y = op_desc.Input("Y")[0];
      if (parameters.count(Y)) {
        it = Registry<OpConverter>::Lookup("fc");
      }
    }
    if (!it) {
      it = Registry<OpConverter>::Lookup(op_desc.Type());
    }
    PADDLE_ENFORCE_NOT_NULL(it, "no OpConverter for optype [%s]",
                            op_desc.Type());
    it->SetEngine(engine);
    (*it)(op, scope);
  }

  // convert fluid block to tensorrt network
  void ConvertBlock(const framework::proto::BlockDesc& block,
                    const std::unordered_set<std::string>& parameters,
                    const framework::Scope& scope, TensorRTEngine* engine) {
    for (int i = 0; i < block.ops_size(); i++) {
      const auto& op = block.ops(i);
      ConvertOp(op, parameters, scope, engine);
    }
  }

  void SetEngine(TensorRTEngine* engine) { engine_ = engine; }

  virtual ~OpConverter() {}

  // TensorRT engine
  TensorRTEngine* engine_{nullptr};

 private:
  // registered op converter map, whose key is the fluid op type, and value is
  // the pointer position of corresponding OpConverter class.
  std::unordered_map<std::string, OpConverter*> converters_;
  // fluid inference scope
  framework::Scope* scope_{nullptr};
};

#define REGISTER_TRT_OP_CONVERTER(op_type__, Converter__)       \
  struct trt_##op_type__##_converter {                          \
    trt_##op_type__##_converter() {                             \
      Registry<OpConverter>::Register<Converter__>(#op_type__); \
    }                                                           \
  };                                                            \
  trt_##op_type__##_converter trt_##op_type__##_converter__;

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
