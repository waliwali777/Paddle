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

#include "paddle/pten/kernels/conj_kernel.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/conj_impl.h"
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;
PT_REGISTER_CTX_KERNEL(conj,
                       CPU,
                       ALL_LAYOUT,
                       pten::Conj,
                       complex64,
                       complex128,
                       float,
                       double,
                       int,
                       int64_t) {}
