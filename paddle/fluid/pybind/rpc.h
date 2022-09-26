//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindServiceInfo(py::module* m);
void BindFuture(py::module* m);
void InitAndSetAgentInstance(py::module* m);
void InvokeRpc(py::module* m);
void StartServer(py::module* m);
void StartClient(py::module* m);
void StopServer(py::module* m);
void GetServiceInfo(py::module* m);
void GetServiceInfoByRank(py::module* m);
void GetCurrentServiceInfo(py::module* m);
void GetAllServiceInfos(py::module* m);
void ClearPythonRpcHandler(py::module* m);
}  // namespace pybind
}  // namespace paddle
