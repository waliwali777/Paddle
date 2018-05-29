/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/grpc_service.h"
#include "paddle/fluid/operators/detail/request_handler.h"
#include "paddle/fluid/operators/detail/rpc_server.h"
#include "paddle/fluid/operators/detail/send_recv.grpc.pb.h"
#include "paddle/fluid/operators/detail/send_recv.pb.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace detail {

class RequestBase;

class AsyncGRPCServer final : public RPCServer {
 public:
  explicit AsyncGRPCServer(const std::string& address, int client_num,
                           int thread_num = 10)
      : RPCServer(address, client_num, thread_num), ready_(0) {}

  virtual ~AsyncGRPCServer() {}
  void WaitServerReady() override;
  void StartServer() override;

 protected:
  void HandleRequest(::grpc::ServerCompletionQueue* cq, int rpc_id,
                     std::function<void(int)> TryToRegisterNewOne);
  void TryToRegisterNewSendOne(int req_id);
  void TryToRegisterNewGetOne(int req_id);
  void TryToRegisterNewPrefetchOne(int req_id);
  void ShutdownQueue();
  void ShutDownImpl() override;

 private:
  static const int kSendReqsBufSize = 100;
  static const int kGetReqsBufSize = 100;
  static const int kPrefetchReqsBufSize = 10;

  std::mutex cq_mutex_;
  volatile bool is_shut_down_ = false;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_send_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_get_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_prefetch_;

  RequestBase* send_reqs_[kSendReqsBufSize];
  RequestBase* get_reqs_[kGetReqsBufSize];
  RequestBase* prefetch_reqs_[kPrefetchReqsBufSize];

  GrpcService::AsyncService service_;
  std::unique_ptr<::grpc::Server> server_;

  // condition of the sub program
  std::mutex barrier_mutex_;
  mutable int barrier_cond_step_;
  std::condition_variable barrier_condition_;

  std::vector<std::unique_ptr<std::thread>> t_sends_;
  std::vector<std::unique_ptr<std::thread>> t_gets_;
  std::vector<std::unique_ptr<std::thread>> t_prefetchs_;

  std::unique_ptr<std::thread> t_prefetch_;

  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;

  int ready_;
};

};  // namespace detail
};  // namespace operators
};  // namespace paddle
