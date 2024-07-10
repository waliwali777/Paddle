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

#include "paddle/phi/kernels/selected_rows/impl/lookup_table_impl.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"

PD_REGISTER_KERNEL(lookup_table_grad_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::LookupTableGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(lookup_table_sparse_grad_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::LookupTableSparseGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
