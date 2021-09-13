#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid.core as core


def monkey_patch_eagertensor():
    def __str__(self):
        from paddle.tensor.to_string import eager_tensor_to_string
        return eager_tensor_to_string(self)

    for method_name, method in (("__str__", __str__)):
        setattr(core.eager.EagerTensor, method_name, method)
