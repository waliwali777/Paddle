//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

class NCCLOpHandleBase : public OpHandleBase {
 public:
  NCCLOpHandleBase(ir::Node *node, const platform::NCCLContextMap *nccl_ctxs)
      : OpHandleBase(node), nccl_ctxs_(nccl_ctxs) {}
  virtual ~NCCLOpHandleBase() {}
  virtual void SetNCCLContextMap(const platform::NCCLContextMap *ctxs) = 0;

 protected:
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  const platform::NCCLContextMap *nccl_ctxs_;
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
