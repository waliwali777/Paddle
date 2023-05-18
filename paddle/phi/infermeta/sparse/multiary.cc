/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/sparse/multiary.h"

namespace phi {
namespace sparse {

void FusedAttentionInferMeta(const MetaTensor& query UNUSED,
                             const MetaTensor& key UNUSED,
                             const MetaTensor& value UNUSED,
                             const MetaTensor& sparse_mask UNUSED,
                             const MetaTensor& key_padding_mask UNUSED,
                             const MetaTensor& attn_mask UNUSED,
                             MetaTensor* out UNUSED,
                             MetaTensor* softmax UNUSED) {
  // TODO(zhouwei,zhangkaihuo) add correct infer meta
}

}  // namespace sparse
}  // namespace phi
