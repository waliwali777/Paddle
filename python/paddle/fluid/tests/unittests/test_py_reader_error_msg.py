# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import unittest
import numpy as np


class TestPyReaderErrorMsg(unittest.TestCase):
    def test_check_input_array(self):
        fluid.reader.GeneratorLoader._check_input_array([
            np.random.randint(
                100, size=[2]), np.random.randint(
                    100, size=[2]), np.random.randint(
                        100, size=[2])
        ])
        self.assertRaises(
            TypeError,
            fluid.reader.GeneratorLoader._check_input_array, [
                np.random.randint(
                    100, size=[2]), np.random.randint(
                        100, size=[1]), np.random.randint(
                            100, size=[3])
            ])


if __name__ == '__main__':
    """The framework of Paddle 2.0 is dynamic graph mode by default, but
     Unittest is implemented based on static graph mode.
     Here is a simple conversion from dygraph to static, and Unittest 
     needs to be modified later."""
    import paddle
    paddle.enable_static()
    unittest.main()
