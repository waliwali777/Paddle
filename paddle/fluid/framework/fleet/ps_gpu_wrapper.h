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

#pragma once

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_CUDA)
#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_box/optimizer/optimizer.cuh"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_resource.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_ps.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

class GpuTask {
 public:
  Scope* scope_{nullptr};
  vector<vector<FeatureKey>> feature_keys_;
  vector<vector<FeatureValue>> feature_values_;
  void BuildTask(int table_id, std::vector<std::unordered_map<uint64_t, std::vector<float>>>& table_map) {
    return;
  }
  uint64_t size() {
    uint64_t total_size = 0;
    for (auto& keys : feature_keys_) {
      total_size += keys.size();
    }
    return total_size;
  }
}

class PSGPUWrapper {
 public:
  virtual ~PSGPUWrapper() {}

  PSGPUWrapper() {}

  void PullSparse(const paddle::platform::Place& place,
                  const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const int table_id,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size);
  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len);
  void CopyForPull(const paddle::platform::Place& place, uint64_t** gpu_keys,
                   const std::vector<float*>& values, void* total_values_gpu,
                   const int64_t* gpu_len, const int slot_num,
                   const int hidden_size, const int expand_embed_dim,
                   const int64_t total_length);

  void CopyForPush(const paddle::platform::Place& place,
                   const std::vector<const float*>& grad_values,
                   void* total_grad_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size, const int expand_embed_dim,
                   const int64_t total_length, const int batch_size);
  void BuildGPUPS(const uint64_t table_id, int feature_dim);
  // PSGPUWrapper singleton
  static std::shared_ptr<PSGPUWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PSGPUWrapper());
    }
    return s_instance_;
  }
  std::vector<std::unordered_map<uint64_t, std::vector<float>>>& GetLocalTable(int table_id) {
    return local_tables_[table_id];
  }

 private:
  static std::shared_ptr<PSGPUWrapper> s_instance_;
  std::unordered_map<uint64_t, std::vector<std::unordered_map<uint64_t,std::vector<float>>>> local_tables_;
  std::shared_ptr<GpuPs<FeatureKey, FeatureValue, FeaturePushValue>> GpuPs_;
  std::sharec_ptr<GpuTask> GpuTask_;
  std::vector<LoDTensor> keys_tensor;  // Cache for pull_sparse
  Optimizer<FeatureValue, FeaturePushValue> opt_;
  std::shared_ptr<HeterBoxResource> resource_;



 protected:
  static bool is_initialized_;

};

}  // end namespace framework
}  // end namespace paddle
#endif
