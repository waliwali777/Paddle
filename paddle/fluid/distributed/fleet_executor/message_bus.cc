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

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/platform/gen_comm_id_helper.cc"  //NOLINT
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace distributed {

void MessageBus::Init(
    const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank,
    const std::unordered_map<int64_t, std::string>& rank_to_addr,
    const std::string& addr, const int cur_rank) {
  PADDLE_ENFORCE_EQ(is_init_, false, platform::errors::AlreadyExists(
                                         "MessageBus is already init."));
  is_init_ = true;
  interceptor_id_to_rank_ = interceptor_id_to_rank;
  rank_to_addr_ = rank_to_addr;
  addr_ = addr;
  cur_rank_ = cur_rank;

  ListenPort();

  std::call_once(once_flag_, []() {
    std::atexit([]() { MessageBus::Instance().Release(); });
  });
}

bool MessageBus::IsInit() const { return is_init_; }

void MessageBus::Release() {
  VLOG(3) << "Message bus releases resource.";
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
  server_.Stop(1000);
  server_.Join();
#endif
}

bool MessageBus::Send(const InterceptorMessage& interceptor_message) {
  // called by Interceptor, send InterceptorMessage to dst
  int64_t src_id = interceptor_message.src_id();
  int64_t dst_id = interceptor_message.dst_id();
  if (IsSameRank(src_id, dst_id)) {
    VLOG(3) << "Send a message from interceptor " << src_id
            << " to interceptor " << dst_id << ", which are in the same ranks.";
    return SendIntraRank(interceptor_message);
  } else {
    VLOG(3) << "Send a message from interceptor " << src_id
            << " to interceptor " << dst_id
            << ", which are in different ranks.";
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
    int retry_time = 0;  // message bus will retry sending for 10 times
    while (retry_time < 10) {
      ++retry_time;
      if (SendInterRank(interceptor_message)) {
        VLOG(3) << "Message bus sends inter rank successfully with "
                << retry_time << " times retries.";
        return true;
      }
    }
    VLOG(3) << "Message bus sends inter rank fail after 10 times retries.";
    return false;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Fleet executor does not support sending message between different "
        "ranks when Paddle is compiled with npu or "
        "isn't compiled with distributed for now."));
#endif
  }
  return true;
}

void MessageBus::ListenPort() {
  if (addr_ == "") {
    VLOG(3) << "No need listen to port since training on single card.";
    return;
  }
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
  // function keep listen the port and handle the message
  InterceptorMessageServiceImpl interceptor_message_service;
  PADDLE_ENFORCE_EQ(server_.AddService(&interceptor_message_service,
                                       brpc::SERVER_DOESNT_OWN_SERVICE),
                    0, platform::errors::Unavailable(
                           "Message bus: init brpc service error."));
  auto addr = paddle::string::Split(addr_, ':');
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", addr_));
  std::string ip = addr[0];
  int port = std::stoi(addr[1]);
  brpc::ServerOptions options;
  options.idle_timeout_sec = -1;
  // start the server
  if (server_.Start(addr_.c_str(), &options) == 0) {
    VLOG(3) << "Message is going to use ip:port: " << addr_ << ".";
    // NOTE: this if branch is used for ut, no need for comm the port. For real
    // scenario, the addr must have been used by gen_comm_id.
  } else {
    while (true) {
      port += 8;  // each time increase the port by 8 (# of gpus)
      std::string new_addr = ip + ':' + std::to_string(port);
      if (server_.Start(new_addr.c_str(), &options) == 0) {
        VLOG(3) << "Message is going to use ip:port: " << new_addr << ".";
        UpdateAddr(new_addr, port);
        addr_ = new_addr;
        break;
      }
    }
  }
  VLOG(3) << "Message bus's listen port thread starts successful on address: "
          << addr_ << ".";
#else
  VLOG(3) << "Fleet executor's ListenPort() is a fake function when Paddle is "
             "compiled with npu or Paddle isn't compiled "
             "with distributed for now.";
#endif
}

bool MessageBus::IsSameRank(int64_t src_id, int64_t dst_id) {
  // check whether the dst is the same rank or different rank with src
  const auto& src_rank = interceptor_id_to_rank_.find(src_id);
  const auto& dst_rank = interceptor_id_to_rank_.find(dst_id);
  PADDLE_ENFORCE_NE(
      src_rank, interceptor_id_to_rank_.end(),
      platform::errors::NotFound(
          "Cannot find rank for src interceptor id %lld. Init error.", src_id));
  PADDLE_ENFORCE_NE(
      dst_rank, interceptor_id_to_rank_.end(),
      platform::errors::NotFound(
          "Cannot find rank for dst interceptor id %lld. Init error.", dst_id));
  if (addr_ == "") {
    // single card training, must be same rank
    return true;
  }
  const auto& src_ip = rank_to_addr_.find(src_rank->second);
  PADDLE_ENFORCE_NE(src_ip, rank_to_addr_.end(),
                    platform::errors::NotFound(
                        "Cannot find addr for src rank id %lld. Init error.",
                        src_rank->second));
  PADDLE_ENFORCE_EQ(
      src_ip->second, addr_,
      platform::errors::Fatal("The src interceptor's addr is %s, while the "
                              "message bus's addr is %s, which are different. "
                              "Init error.",
                              src_ip->second, addr_));
  return src_rank->second == dst_rank->second;
}

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
bool MessageBus::SendInterRank(const InterceptorMessage& interceptor_message) {
  // send the message inter rank (dst is different rank with src)
  int64_t dst_id = interceptor_message.dst_id();
  int64_t dst_rank = interceptor_id_to_rank_[dst_id];
  auto dst_ip = rank_to_addr_.find(dst_rank);
  PADDLE_ENFORCE_NE(dst_ip, rank_to_addr_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find rank for dst interceptor id %lld. "
                        "Init error.",
                        dst_id));
  VLOG(3) << "Message bus sending to addr: " << dst_ip->second;
  const char* dst_ip_for_brpc = dst_ip->second.c_str();
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connect_timeout_ms = 1000;
  options.timeout_ms = 1000;
  options.max_retry = 5;
  PADDLE_ENFORCE_EQ(
      channel.Init(dst_ip_for_brpc, &options), 0,
      platform::errors::Unavailable("Message bus: init brpc channel error."));
  TheInterceptorMessageService_Stub stub(&channel);
  InterceptorResponse response;
  brpc::Controller ctrl;
  ctrl.set_log_id(0);
  stub.InterceptorMessageService(&ctrl, &interceptor_message, &response, NULL);
  if (!ctrl.Failed()) {
    if (response.rst()) {
      VLOG(3) << "Message bus: brpc sends success.";
      return true;
    } else {
      VLOG(4) << "Message bus: InterceptorMessageService error.";
      return false;
    }
  } else {
    VLOG(4) << "Message bus: brpc sends failed with error text: "
            << ctrl.ErrorText();
    return false;
  }
}

struct UpdateAddress {
  int rank = -1;
  int port = -1;
};

void MessageBus::UpdateAddr(const std::string& new_addr, int port) {
  int64_t nranks = rank_to_addr_.size();
  VLOG(3) << cur_rank_
          << "' message bus is broadcasting it's new addr: " << new_addr << ".";
  UpdateAddress update_address;
  update_address.rank = cur_rank_;
  update_address.port = port;
  char buffer[MAX_COMMUNIQUEID_LEN] = {0};
  memcpy(buffer, &update_address, sizeof(UpdateAddress));

  int server = paddle::platform::CreateListenSocket(addr_);
  VLOG(3) << "Message bus created a socket to listen address: " << addr_ << ".";
  for (int i = 0; i < cur_rank_; ++i) {
    // update the addresses for ranks before cur rank
    ReceiveANewAddress(server);
  }

  VLOG(3) << "Sending new address to all peers.";
  for (const auto& pair : rank_to_addr_) {
    // establish socket connects with peers
    if (pair.first == cur_rank_) {
      continue;
    }
    std::string ep = pair.second;
    VLOG(3) << "Message bus is connecting endpoint: " << ep << ".";
    paddle::platform::CommHead fake_head;
    int conn = paddle::platform::ConnectAddr(ep, fake_head);
    VLOG(3) << "Connecting finished.";

    CHECK_SYS_CALL(
        paddle::platform::SocketSend(conn, buffer, sizeof(UpdateAddress)),
        "Send new addr.");
    paddle::platform::CloseSocket(conn);
  }
  VLOG(3) << "Finish sending.";

  for (int i = cur_rank_ + 1; i < nranks; ++i) {
    // update the addresses for ranks behind cur rank
    ReceiveANewAddress(server);
  }

  std::stringstream ss;
  ss << "\nThe DNS table of the message bus after updating is: \n";
  for (const auto& pair : rank_to_addr_) {
    ss << pair.first << "\t->\t" << pair.second << "\n";
  }
  VLOG(5) << ss.str();
}

void MessageBus::ReceiveANewAddress(int server) {
  char buffer[MAX_COMMUNIQUEID_LEN] = {0};

  CHECK_SYS_CALL(
      paddle::platform::SocketRecv(server, &buffer, sizeof(UpdateAddress)),
      "Receive new addr.");
  UpdateAddress received;
  memcpy(&received, buffer, sizeof(UpdateAddress));
  VLOG(3) << "Update address for rank: " << received.rank
          << ". The new port for it is: " << received.port << ".";
  auto iter = rank_to_addr_.find(received.rank);
  PADDLE_ENFORCE_NE(
      iter, rank_to_addr_.end(),
      platform::errors::InvalidArgument(
          "Message bus received an unknown rank: %d.", received.rank));
  std::string old_ip =
      paddle::string::Split(rank_to_addr_[received.rank], ':')[0];
  std::string new_addr = old_ip + ':' + std::to_string(received.port);
  VLOG(3) << "The new address for rank: " << received.rank << " is " << new_addr
          << ".";
  rank_to_addr_[received.rank] = new_addr;
}
#endif

bool MessageBus::SendIntraRank(const InterceptorMessage& interceptor_message) {
  // send the message intra rank (dst is the same rank with src)
  return Carrier::Instance().EnqueueInterceptorMessage(interceptor_message);
}

}  // namespace distributed
}  // namespace paddle
