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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.framework import core
from paddle.fluid.dygraph.base import switch_to_static_graph

paddle.enable_static()

class TestTakeAlongAxisOp(OpTest):
    def setUp(self):
        self.init_data()
        self._cpu_only = True
        self.dtype = 'float64'
        self.op_type = "take_along_axis"
        self.inputs = {
            'Input': np.array(self.xnp).astype(self.x_type),
            'Index': np.array(self.index).astype(self.index_type),
            }
        self.attrs = {
            'Dim': self.dim
            }
        self.target = np.take_along_axis(np.array(self.xnp), np.array(self.index), self.dim)
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass

        #self.check_grad(['Input'], 'Result')

    def init_data(self):
        self.xnp = [[1, 2], [3, 4]]
        print(">>>self.xnp>>>", self.xnp)
        self.x_type = "float64"
        self.index = [[0, 0], [1, 0]]
        self.index_type = "int32"
        self.dim = 1
        self.dim_type = "int64"


# class TestCase1(TestTakeAlongAxisOp):
#     def init_data(self):
#         self.xnp =[[1,20,30], [40,50,60]]
#         self.x_type = "float64"
#         self.index = [[1,2]]
#         self.index_type = "int32"
#         self.dim = 1
#         self.dim_type = "int64"

# class TestCase2(TestTakeAlongAxisOp):
#     def init_data(self):
#         self.x_shape = (10, 20)
#         self.xnp = np.random.random(self.x_shape)
#         self.x_type = "float64"
#         self.index = [[1,2]]
#         self.index_type = "int32"
#         self.dim = 1
#         self.dim_type = "int64"

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
