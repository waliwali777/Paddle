// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"

class GradNodematmul_v2 : public egr::GradNodeBase {
 public:
  GradNodematmul_v2() : egr::GradNodeBase() {}
  GradNodematmul_v2(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodematmul_v2() override = default;

  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      const std::vector<std::vector<paddle::experimental::Tensor>>& grads)
      override;

  // SetX, SetY, ...
  void SetTensorWrapperX(const paddle::experimental::Tensor& X) { X_ = X; }
  void SetTensorWrapperY(const paddle::experimental::Tensor& Y) { Y_ = Y; }

  // SetAttr0, SetAttr1, ...
  void SetAttrmkldnn_data_type(const std::string& mkldnn_data_type) {
    mkldnn_data_type_ = mkldnn_data_type;
  }
  void SetAttruse_mkldnn(const bool use_mkldnn) { use_mkldnn_ = use_mkldnn; }
  void SetAttrtrans_x(const bool trans_x) { trans_x_ = trans_x; }
  void SetAttrtrans_y(const bool trans_y) { trans_y_ = trans_y; }

 private:
  // TensorWrappers
  paddle::experimental::Tensor X_;
  paddle::experimental::Tensor Y_;

  // Attribute Members
  std::string mkldnn_data_type_ = "float32";
  bool use_mkldnn_ = 0;
  bool trans_x_ = 0;
  bool trans_y_ = 0;
};
