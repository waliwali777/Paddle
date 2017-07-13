#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

void PlainNet::AddOp(const OpDesc& desc) {
  OpPtr op(OpRegistry::CreateOp(desc));
  ops_.push_back(std::move(op));
}

void PlainNet::InferShape(const std::shared_ptr<Scope>& scope) const {
  for (auto& op : ops_) {
    op->InferShape(scope);
  }
}

void PlainNet::Run(ScopePtr scope, DeviceContext* ctx) {
  for (auto& op : ops_) {
    op->Run(scope, dev_ctx);
  }
}

// REGISTER_OP(plainnet_operator, PlainNet, PlainNetOpProtoAndCheckerMaker);
}  // namespace framework
}  // namespace paddle
