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

#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

static inline Tensor *GetTensorFromVar(Variable *var) {
  // TODO(zjl): support SelectedRows
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else {
    PADDLE_THROW("Variable must be type of LoDTensor");
  }
}

ShareTensorBufferOpHandle::ShareTensorBufferOpHandle(
    ir::Node *node, const Scope *scope, size_t scope_idx,
    const std::string &op_type,
    const std::vector<ir::MemOptVarInfo *> &in_var_infos,
    const std::vector<std::string> &out_var_names)
    : OpHandleBase(node),
      scope_(scope),
      scope_idx_(scope_idx),
      op_type_(op_type),
      in_var_infos_(in_var_infos),
      out_var_names_(out_var_names) {
  for (auto &in_var_info : in_var_infos_) {
    PADDLE_ENFORCE_NOT_NULL(in_var_info, "in_var_info cannot be nullptr");
  }
  PADDLE_ENFORCE_EQ(in_var_infos_.size(), out_var_names_.size());
}

std::unordered_set<std::string> ShareTensorBufferOpHandle::ReusedVarSet()
    const {
  std::unordered_set<std::string> result;
  for (auto &in_var_info : in_var_infos_) {
    result.insert(in_var_info->Name());
  }
  return result;
}

void ShareTensorBufferOpHandle::Add(ir::MemOptVarInfo *in_var_info,
                                    const std::string &out_var_name) {
  PADDLE_ENFORCE_NOT_NULL(in_var_info);
  in_var_infos_.emplace_back(in_var_info);
  out_var_names_.emplace_back(out_var_name);
}

void ShareTensorBufferOpHandle::InitCUDA() {
#ifdef PADDLE_WITH_CUDA
  int dev_id =
      boost::get<platform::CUDAPlace>(dev_ctxes_.begin()->first).device;
  events_[dev_id] = nullptr;
#endif
}

void ShareTensorBufferOpHandle::CallOnce() {
  PADDLE_ENFORCE(in_out_vars_.empty(), "in_out_vars_ must be initialized here");
  Scope *exec_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
  for (size_t i = 0; i < in_var_infos_.size(); ++i) {
    auto *in_var = exec_scope->FindVar(in_var_infos_[i]->Name());
    auto *out_var = exec_scope->FindVar(out_var_names_[i]);
    PADDLE_ENFORCE_NOT_NULL(in_var);
    PADDLE_ENFORCE_NOT_NULL(out_var);
    in_out_vars_.emplace_back(in_var, out_var);
  }
  is_shared_.assign(in_var_infos_.size(), false);
}

void ShareTensorBufferOpHandle::RunImpl() {
  if (in_var_infos_.size() != in_out_vars_.size()) {
    CallOnce();
  }

  for (size_t i = 0; i < in_var_infos_.size(); ++i) {
    auto in_var_info = in_var_infos_[i];
    if (UNLIKELY(in_var_info->IsSkipped())) {
      // If in_var is inplaced in the previous batch and we want to fetch
      // in_var in the current batch, we have to reset memory of out_var
      // to avoid wrong calcualtion result.
      if (is_shared_[i]) {
        auto *out_tensor = GetTensorFromVar(in_out_vars_[i].second);
        VLOG(1) << "Clear " << out_var_names_[i]
                << " because you may want to fetch an inplaced variable "
                << in_var_info->Name()
                << " in previous batch: " << in_var_info->Name() << " -> "
                << out_var_names_[i];
        out_tensor->clear();
        is_shared_[i] = false;
      }
    } else {
      auto *in_tensor = GetTensorFromVar(in_out_vars_[i].first);
      auto *out_tensor = GetTensorFromVar(in_out_vars_[i].second);
      out_tensor->ShareBufferWith(*in_tensor);

      VLOG(2) << "Share tensor buffer when running " << op_type_ << " : "
              << in_var_info->Name() << " -> " << out_var_names_[i];

      is_shared_[i] = true;
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
