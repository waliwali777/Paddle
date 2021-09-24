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

#include "paddle/fluid/framework/cinn/cinn_runner.h"

#include <map>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {
namespace cinn {

using ir::Graph;

std::map<std::string, FetchType*> CinnRunner::Run(
    const Graph& graph, Scope* scope,
    std::map<std::string, const LoDTensor*>* feed_targets) {
  CinnCacheKey cur_key(graph, *feed_targets);
  std::shared_ptr<CinnCompiledObject> obj_to_run;
  if (cache_.find(cur_key) != cache_.end()) {
    obj_to_run = cache_[cur_key];
  } else {
    obj_to_run = std::make_shared<CinnCompiledObject>();
    obj_to_run->Compile(graph, feed_targets);
    cache_[cur_key] = obj_to_run;
  }
  return obj_to_run->Run(scope, feed_targets);
}

}  // namespace cinn
}  // namespace framework
}  // namespace paddle
