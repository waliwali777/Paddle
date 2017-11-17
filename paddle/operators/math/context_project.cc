/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/context_project.h"

namespace paddle {
namespace operators {
namespace math {

template class ContextProjectFunctor<platform::CPUPlace, float>;
template class ContextProjectFunctor<platform::CPUPlace, double>;
template class ContextProjectFunctor<platform::CPUPlace, int>;
template class ContextProjectFunctor<platform::CPUPlace, int64_t>;
template class ContextProjectFunctor<platform::CPUPlace, bool>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
