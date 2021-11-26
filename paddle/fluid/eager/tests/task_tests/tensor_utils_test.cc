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

#include <sstream>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/pten/api/lib/utils/allocator.h"

#include "paddle/pten/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

namespace eager_test {

TEST(TensorUtils, Test) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<egr::EagerTensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  egr::EagerTensor t = CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 5.0 /*value*/, true /*is_leaf*/);

  egr::EagerTensor t_grad = CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 1.0 /*value*/, false /*is_leaf*/);

  CHECK_EQ(IsLeafTensor(t), true);

  // Test Utils
  CompareTensorWithValue<float>(t, 5.0);

  egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(&t);
  *meta->MutableGrad() = t_grad;

  CompareGradTensorWithValue<float>(t, 1.0);
}

}  // namespace eager_test
