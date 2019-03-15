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
#include "paddle/fluid/framework/details/op_handle_base.h"
#include <map>

namespace paddle {
namespace framework {
namespace details {
std::string OpHandleBase::DebugString() const {
  std::stringstream ss;
  ss << "(";
  for (auto *var : inputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ") --> (";
  for (auto *var : outputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ")\n";
  return ss.str();
}

OpHandleBase::~OpHandleBase() {
#ifdef PADDLE_WITH_CUDA
  for (auto &ev : events_) {
    PADDLE_ENFORCE(cudaEventDestroy(ev.second));
  }
#endif
}

void OpHandleBase::Run(bool use_cuda) {
#ifdef PADDLE_WITH_CUDA
  if (events_.empty() && use_cuda) {
    for (auto &p : dev_ctxes_) {
      int dev_id = boost::get<platform::CUDAPlace>(p.first).device;
      PADDLE_ENFORCE(cudaSetDevice(dev_id));
      PADDLE_ENFORCE(
          cudaEventCreateWithFlags(&events_[dev_id], cudaEventDisableTiming));
    }
    if (dev_ctxes_.size() > 0) {
      if (IsMultiDeviceTransfer()) {
        // TODO(zcd): Get the relationshape of output and place
        for (auto &var : outputs_) {
          // The DummyVarHandle is only used to represent dependencies between
          // operators,
          auto *var_handle = dynamic_cast<VarHandle *>(var);
          if (var_handle == nullptr) continue;
          auto &var_place = var_handle->place();
          int dev_id = boost::get<platform::CUDAPlace>(var_place).device;
          var_handle->SetGenerateEvent(events_[dev_id]);
        }
      } else {
        PADDLE_ENFORCE_EQ(dev_ctxes_.size(), 1,
                          "OpHandle(%s)'s should only have one dev_ctx.",
                          Name());
        auto &place = dev_ctxes_.begin()->first;
        int dev_id = boost::get<platform::CUDAPlace>(place).device;
        for (auto &var : outputs_) {
          // The DummyVarHandle is only used to represent dependencies between
          // operators,
          auto *var_handle = dynamic_cast<VarHandle *>(var);
          if (var_handle == nullptr) continue;
          PADDLE_ENFORCE(
              platform::is_same_place(var_handle->place(), place),
              "The place of output VarHandle and OpHandle is not equal.");
          var_handle->SetGenerateEvent(events_[dev_id]);
        }
      }
    } else {
      VLOG(3) << string::Sprintf("OpHandle(%s)'s doesn't have dev_ctx.",
                                 Name());
    }
  }
#else
  PADDLE_ENFORCE(!use_cuda);
#endif
  RunImpl();
}

void OpHandleBase::RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_NOT_NULL(waited_ctx);
  if (platform::is_cpu_place(waited_ctx->GetPlace()) || events_.empty()) {
    for (auto &dev_ctx : dev_ctxes_) {
      PADDLE_ENFORCE_NOT_NULL(dev_ctx.second);
      dev_ctx.second->Wait();
    }
  } else {
    auto stream =
        static_cast<platform::CUDADeviceContext *>(waited_ctx)->stream();
    for (auto &ev : events_) {
      PADDLE_ENFORCE(cudaStreamWaitEvent(stream, ev.second, 0));
    }
  }
#else
  for (auto &dev_ctx : dev_ctxes_) {
    dev_ctx.second->Wait();
  }
#endif
}

void OpHandleBase::RecordWaitEventOnCtx2(
    const std::vector<VarHandle *> &in_vars,
    platform::DeviceContext *waited_ctx) {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_NOT_NULL(waited_ctx);
  std::unordered_set<cudaEvent_t> generate_input_events;

  for (auto &in_var : in_vars) {
    if (in_var->HasEvent()) {
      generate_input_events.insert(in_var->GetEvent());
    }
  }

  if (platform::is_cpu_place(waited_ctx->GetPlace()) ||
      generate_input_events.empty()) {
    for (auto &dev_ctx : dev_ctxes_) {
      PADDLE_ENFORCE_NOT_NULL(dev_ctx.second);
      dev_ctx.second->Wait();
    }
  } else {
    auto stream =
        static_cast<platform::CUDADeviceContext *>(waited_ctx)->stream();
    for (auto &event : generate_input_events) {
      PADDLE_ENFORCE(cudaStreamWaitEvent(stream, event, 0));
    }
  }
#else
  for (auto &dev_ctx : dev_ctxes_) {
    dev_ctx.second->Wait();
  }
#endif
}

void OpHandleBase::AddInput(VarHandleBase *in) {
  this->inputs_.emplace_back(in);
  node_->inputs.push_back(in->Node());
  in->AddOutput(this, this->Node());
}

void OpHandleBase::AddOutput(VarHandleBase *out) {
  outputs_.emplace_back(out);
  node_->outputs.push_back(out->Node());
  out->AddInput(this, this->Node());
}

size_t OpHandleBase::NoDummyInputSize() const {
  size_t cnt = 0;
  for (auto *in : inputs_) {
    if (dynamic_cast<DummyVarHandle *>(in) == nullptr) {
      ++cnt;
    }
  }
  return cnt;
}

void OpHandleBase::RunAndRecordEvent(const std::function<void()> &callback) {
  callback();
#ifdef PADDLE_WITH_CUDA
  for (auto &p : dev_ctxes_) {
    if (!events_.empty()) {  // Use event
      auto &event = events_.at(boost::get<platform::CUDAPlace>(p.first).device);
      auto stream =
          static_cast<platform::CUDADeviceContext *>(p.second)->stream();
      PADDLE_ENFORCE(cudaEventRecord(event, stream));
    }
  }
#endif
}

void OpHandleBase::RunAndRecordEvent(platform::Place p,
                                     const std::function<void()> &callback) {
  callback();
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(p) && events_.empty()) {
    auto *ctx = dev_ctxes_.at(p);
    auto &event = events_.at(boost::get<platform::CUDAPlace>(p).device);
    auto stream = static_cast<platform::CUDADeviceContext *>(ctx)->stream();
    PADDLE_ENFORCE(cudaEventRecord(event, stream));
  }
#endif
}

size_t OpHandleBase::NotReadyInputSize() const {
  std::unordered_set<VarHandleBase *> res;
  for (auto *var : inputs_) {
    if (var->GeneratedOp() != nullptr) {
      res.emplace(var);
    }
  }
  return res.size();
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
