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

#include "paddle/fluid/framework/fleet/heter_wrapper.h"
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

std::shared_ptr<HeterWrapper> HeterWrapper::s_instance_ = NULL;
bool HeterWrapper::is_initialized_ = false;

void HeterWrapper::CreateClient2XpuConnection() {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connection_type = "single";
  
  xpu_channels_.resize(xpu_list_.size());
  for (size_t i = 0; i < xpu_list_.size(); ++i) {
    VLOG(3) << "channel init: " << xpu_list_[i];
    xpu_channels_[i].reset(new brpc::Channel());
    if (xpu_channels_[i]->Init(xpu_list_[i].c_str(), "", &options) != 0) {
      VLOG(0) << "server channel init fail";
    }
  }
}

void HeterWrapper::RegisterServiceHandler(HeterServiceHandler func) {
  service_.RegisterServiceHandler(func);
}

void HeterWrapper::SetXpuList(const std::vector<std::string>& xpu_list) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to set xpu list";
  for (auto& x : xpu_list) {
    xpu_list_.push_back(x);
    VLOG(3) << "set xpu list:  " << x << " size: " << xpu_list_.size();
  }
#endif
}

void HeterWrapper::StartXpuService(const std::string& ip, uint32_t port) {
  std::string ip_port = ip + ":" + std::to_string(port);
  VLOG(3) << "xpu server starts at " << ip_port;
  
  server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
       
  if (server_.Start(ip_port.c_str(),  &options) != 0) {
      VLOG(0) << "xpu server start fail";
  }
}

//void HeterWrapper::SerializeToReq(const std::string& varname, Scope* scope, HeterRequest& request) {
//  auto* req_var = request.mutable_vars();
  
void HeterWrapper::SerializeToReq(const std::string& varname, Scope* scope, VariableMessage* req_var) {
  Variable* var = scope->FindVar(varname);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  req_var->set_varname(varname); 
  req_var->set_type(LOD_TENSOR);
  req_var->set_data_type(static_cast<VariableMessage::Type>(tensor->type()));

  for (auto& dim : framework::vectorize(tensor->dims())) {
    req_var->add_dims(dim);
  }
  const framework::LoD lod = tensor->lod();
  if (lod.size() > 0) {
    req_var->set_lod_level(lod.size());
    for (auto& each : lod) {
      VariableMessage::LodData* lod_inner = req_var->add_lod();
      for (auto& d : each) {
        lod_inner->add_lod_data(d);
      }
    }
  }

  auto* req_data = req_var->mutable_data();
  req_data->clear();
  req_data->resize(tensor->numel() * SizeOfType(tensor->type()));
  char* data_ptr = const_cast<char*>(req_data->data());

  memcpy(data_ptr, tensor->data<void>(), tensor->numel() * SizeOfType(tensor->type()));
}

//void HeterWrapper::DeSerializeToTensor(Scope* scope, const HeterRequest* request) {
void HeterWrapper::DeSerializeToTensor(Scope* scope, const VariableMessage& req_var) {
  //const VariableMessage& req_var = request->vars();
  auto* var = scope->FindVar(req_var.varname());
  auto* tensor = var->GetMutable<LoDTensor>();

  std::vector<int> vec_dim;
  for (auto& x : req_var.dims()) {
    vec_dim.push_back(x);
  }
  tensor->Resize(make_ddim(vec_dim));

  LoD lod;
  for (int i = 0; i < req_var.lod_level(); ++i) {
    framework::Vector<size_t> v;
    for (int j = 0; j < req_var.lod(i).lod_data_size(); ++j) {
      v.push_back(req_var.lod(i).lod_data(j));
    }
    lod.push_back(v);
  }
  tensor->set_lod(lod);

  void* tensor_data =
      tensor->mutable_data(platform::CPUPlace(), ToVarType(req_var.data_type()));
  memcpy(tensor_data, req_var.data().data(), tensor->numel() * SizeOfType(tensor->type()));
}

framework::proto::VarType::Type HeterWrapper::ToVarType(
    VariableMessage::Type type) {
  switch (type) {
    case VariableMessage::FP32:
      return framework::proto::VarType::FP32;  // NOLINT
    case VariableMessage::FP64:
      return framework::proto::VarType::FP64;  // NOLINT
    case VariableMessage::INT32:
      return framework::proto::VarType::INT32;  // NOLINT
    case VariableMessage::INT64:
      return framework::proto::VarType::INT64;  // NOLINT
    case VariableMessage::BOOL:
      return framework::proto::VarType::BOOL;  // NOLINT
    default:
      PADDLE_THROW("Not support type %d", type);
  }
}


//void HeterWrapper::CallRemoteXpu(Scope* scope, int cur_batch) {
//  HeterRequest request;
//  HeterResponse response;
//  brpc::Controller cntl;
//  request.set_cmd(0);
//  request.set_cur_batch(cur_batch);
//  
//  std::vector<std::string> varnames = {"concat_1.tmp_0", "click", "12345"};
//  for (auto& varname : varnames) {
//    auto* req_var = request.add_vars();
//    SerializeToReq(varname, scope, req_var);
//  }
//  
//  HeterService_Stub stub(xpu_channels_[0].get());
//  stub.service(&cntl, &request, &response, NULL);
//  if (cntl.Failed()) {
//    VLOG(0) << "call xpu fail: " << cntl.ErrorText();
//  }
//  else {
//    VLOG(3) << "call xpu success";
//  }
//  
//  DeSerializeToTensor(scope, response.vars());
//}


void HeterWrapper::CallRemoteXpu(std::shared_ptr<HeterTask> task, HeterCpuWorker* worker) {
  HeterRequest request;
  request.set_cmd(0);
  request.set_cur_batch(task->cur_batch_);
  
  OnHeterRpcDone *done = new OnHeterRpcDone([this, task, worker] (void* done) {
    auto* closure = (OnHeterRpcDone*)done;
    if (closure->cntl.Failed()) {
      VLOG(0) << "call xpu fail: " << closure->cntl.ErrorText();
    }
    else {
      VLOG(3) << "call xpu success";
    }
    DeSerializeToTensor(task->scope_, closure->response.vars());

    worker->Schedule(task->taskid_); 
  }); 

  std::vector<std::string> varnames = {"concat_1.tmp_0", "click", "12345"};
  for (auto& varname : varnames) {
    auto* req_var = request.add_vars();
    SerializeToReq(varname, task->scope_, req_var);
  }
  
  HeterService_Stub stub(xpu_channels_[0].get());
  //stub.service(&cntl, &request, &response, brpc::NewCallback(&HeterWrapper::RpcCallBack, response, cntl, worker, task));
  stub.service(&done->cntl, &request, &done->response, done);
  
}

}  // end namespace framework
}  // end namespace paddle
