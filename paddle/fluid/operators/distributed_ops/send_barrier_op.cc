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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/distributed.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

namespace distributed {
class RPCClient;
}  // namespace distributed

class SendBarrierOp : public framework::OperatorBase {
 public:
  SendBarrierOp(const std::string& type,
                const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto is_half_async = Attr<bool>("half_async");

    if (is_half_async) {
      distributed::Communicator::GetInstance()->Barrier();
      return;
    }

    std::vector<std::string> eps = Attr<std::vector<std::string>>("endpoints");

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(
            Attr<int>("trainer_id"));

    VLOG(3) << "SendBarrierOp sync";

    std::vector<distributed::VarHandlePtr> rets;

    for (auto& ep : eps) {
      VLOG(3) << "send barrier, ep: " << ep;
      rets.push_back(rpc_client->AsyncSendBatchBarrier(ep));
    }

    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE_NE(
          rets[i]->Wait(), 0U,
          platform::errors::ExecutionTimeout("internal error in RPCClient"));
    }
  }
};

class SendBarrierOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Any) Dummy inputs, used for control dependency")
        .AsDuplicable();
    AddOutput("Out", "(Any) Dummy outputs, used for control dependency")
        .AsDuplicable();
    AddComment(R"DOC(
SendBarrier operator

This operator will send a send barrier signal to list_and_serv op, so that
the Parameter Server would knew all variables have been sent.
)DOC");

    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<bool>(
        "half_async",
        "(bool, default false)"
        "half_async=True is for half_async mode, this will send signal "
        "to HalfAsyncCommunicator Instance")
        .SetDefault(false);
  }
};

class SendBarrierOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    send_barrier, ops::SendBarrierOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::SendBarrierOpMaker, ops::SendBarrierOpShapeInference);
