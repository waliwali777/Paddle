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

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include <string>

namespace paddle {
namespace framework {
namespace details {

ComputationOpHandle *GetUniquePendingComputationOpHandle(
    ShareTensorBufferOpHandle *share_tensor_op) {
  ComputationOpHandle *result_op = nullptr;
  for (ir::Node *out_var : share_tensor_op->Node()->outputs) {
    for (ir::Node *pending_op : out_var->outputs) {
      auto &op = pending_op->Wrapper<OpHandleBase>();
      auto *compute_op = dynamic_cast<ComputationOpHandle *>(&op);
      PADDLE_ENFORCE_NOT_NULL(compute_op);

      if (result_op == nullptr) {
        result_op = compute_op;
      } else {
        PADDLE_ENFORCE_EQ(result_op, compute_op);
      }
    }
  }

  PADDLE_ENFORCE_NOT_NULL(result_op);
  return result_op;
}

ComputationOpHandle::ComputationOpHandle(ir::Node *node, Scope *scope,
                                         platform::Place place,
                                         size_t scope_idx)
    : OpHandleBase(node),
      op_(framework::OpRegistry::CreateOp(*node->Op())),
      scope_(scope),
      place_(place),
      scope_idx_(scope_idx) {}

void ComputationOpHandle::RunImpl() {
  WaitInputVarGenerated(place_);

  if (functor_) {
    (*functor_)(local_exec_scopes_[0]);
  }

  auto run_func = [this]() { op_->Run(*local_exec_scopes_[0], place_); };

  if (is_lock_and_record_event_free_) {
    run_func();
  } else {
    this->RunAndRecordEvent(run_func);
  }
}

bool ComputationOpHandle::NeedWait(VarHandleBase *in_var) {
  bool need_wait =
      in_var && in_var->GeneratedOp() &&
      in_var->GeneratedOp()->DeviceContext(place_) != dev_ctxes_.at(place_);
  return need_wait;
}

std::string ComputationOpHandle::Name() const { return op_->Type(); }
}  // namespace details
}  // namespace framework
}  // namespace paddle
