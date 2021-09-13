# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
from paddle.fluid import core
from paddle.device import *


class TestGetDeviceProperties(unittest.TestCase):
    def test_get_device_properties_int(self):
        gpu_num = paddle.device.cuda.device_count()
        for i in range(gpu_num):
            props = paddle.get_device_properties(i)
            self.assertIsNotNone(props)

    def test_get_device_properties_CUDA(self):
        if is_compiled_with_cuda:
            device = core.CUDAPlace(0)
            props = paddle.get_device_properties(device)
            self.assertIsNotNone(props)


if __name__ == "__main__":
    unittest.main()
