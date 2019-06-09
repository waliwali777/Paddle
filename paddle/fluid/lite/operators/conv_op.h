// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class ConvOpLite : public OpLite {
 public:
  ConvOpLite() {}

  explicit ConvOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    auto input = op_desc.Input("Input").front();
    auto filter = op_desc.Input("Filter").front();
    auto out = op_desc.Output("Out").front();
    param_.x = scope->FindVar(input)->GetMutable<lite::Tensor>();
    param_.filter = scope->FindVar(filter)->GetMutable<lite::Tensor>();
    CHECK(scope->FindVar(out));
    param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
    param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    param_.groups = op_desc.GetAttr<int>("groups");
    param_.dilations = op_desc.GetAttr<std::vector<int>>("dilations");
    // optional params
    std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
    if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
        input_arg_names.end()) {
      auto bias_var = scope->FindVar(op_desc.Input("Bias").front());
      if (bias_var != nullptr) {
        param_.bias =
            const_cast<lite::Tensor *>(&(bias_var->Get<lite::Tensor>()));
      }
    }
    if (std::find(input_arg_names.begin(), input_arg_names.end(), "ResidualData") !=
        input_arg_names.end()) {
      auto residual_data_var = scope->FindVar(op_desc.Input("ResidualData").front());
      if (residual_data_var != nullptr) {
        param_.residualData =
            const_cast<lite::Tensor *>(&(residual_data_var->Get<lite::Tensor>()));
      }
    }
    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "conv2d"; }

 private:
  mutable ConvParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
