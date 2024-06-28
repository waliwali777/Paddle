// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature LookupTableGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("W")) {
    if ((paddle::any_cast<bool>(ctx.Attr("is_sparse"))) == true) {
      return KernelSignature("lookup_table_sparse_grad",
                             {"W", "Ids", "Out@GRAD"},
                             {"is_sparse",
                              "is_distributed",
                              "padding_idx",
                              "remote_prefetch",
                              "entry_config",
                              "is_test",
                              "entry",
                              "table_class",
                              "table_names",
                              "trainer_id",
                              "grad_inplace",
                              "epmap",
                              "height_sections"},
                             {"W@GRAD"});
    } else {
      return KernelSignature("lookup_table_grad",
                             {"W", "Ids", "Out@GRAD"},
                             {"is_sparse",
                              "is_distributed",
                              "padding_idx",
                              "remote_prefetch",
                              "entry_config",
                              "is_test",
                              "entry",
                              "table_class",
                              "table_names",
                              "trainer_id",
                              "grad_inplace",
                              "epmap",
                              "height_sections"},
                             {"W@GRAD"});
    }
  } else {
    if ((paddle::any_cast<bool>(ctx.Attr("is_sparse"))) == true) {
      return KernelSignature("lookup_table_sparse_grad_sr",
                             {"W", "Ids", "Out@GRAD"},
                             {"is_sparse",
                              "is_distributed",
                              "padding_idx",
                              "remote_prefetch",
                              "entry_config",
                              "is_test",
                              "entry",
                              "table_class",
                              "table_names",
                              "trainer_id",
                              "grad_inplace",
                              "epmap",
                              "height_sections"},
                             {"W@GRAD"});
    } else {
      return KernelSignature("lookup_table_grad_sr",
                             {"W", "Ids", "Out@GRAD"},
                             {"is_sparse",
                              "is_distributed",
                              "padding_idx",
                              "remote_prefetch",
                              "entry_config",
                              "is_test",
                              "entry",
                              "table_class",
                              "table_names",
                              "trainer_id",
                              "grad_inplace",
                              "epmap",
                              "height_sections"},
                             {"W@GRAD"});
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(lookup_table_grad,
                           phi::LookupTableGradOpArgumentMapping);
