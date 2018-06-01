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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class Conv2dOpConverter : public OpConverter {
 public:
  Conv2dOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope) override {
    LOG(INFO)
        << "convert a fluid conv2d op to tensorrt conv layer without bias";
  }
};

REGISTER_TRT_OP_CONVERTER(conv2d, Conv2dOpConverter);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
