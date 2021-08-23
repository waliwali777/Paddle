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

/* 
    Each Operation has a specific GradNode inheritted from GradNodeBase
    A specific GradNode defines
    1. Input Tensors
    2. overrides operator() to perform actual backward computations

    TODO: Generate GradNode via auto-code-generation
*/
namespace egr {

class GradNodeScale : public GradNodeBase {
 public:
  // Constructor: configure fwd input tensors to grad node
  GradNodeScale() : GradNodeBase(1) {}
  ~GradNodeScale() override = default;

  // Functor: perform backward computations
  virtual std::vector<pt::Tensor> operator()(
      const std::vector<pt::Tensor>& grads) override;
  
  virtual void SetTensorWrappers(const std::vector<pt::Tensor>& tensors) override;
  
  void SetAttributes(float scale);

  // Members: define fwd input tensors
  // For Scale there is no fwd input tensor needed   
 private:
  float scale_ = 1.0;
};

} // namespace egr
