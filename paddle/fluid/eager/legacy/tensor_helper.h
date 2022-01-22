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

#pragma once

#include <vector>
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/pten/api/all.h"
namespace egr {
namespace legacy {

void InitializeVariable(paddle::framework::Variable* var,
                        paddle::framework::proto::VarType::Type var_type);
paddle::framework::proto::VarType::Type GetDtypeFromVar(
    const paddle::framework::Variable& var);
const paddle::platform::Place& GetPlaceFromVar(
    const paddle::framework::Variable& var);
void CopyVariable(const paddle::framework::Variable& src_var,
                  paddle::framework::Variable* dst_var);

}  // namespace legacy
}  // namespace egr
