#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestZerosOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input dtype of zeros_op must be bool, float16, float32, float64, int32, int64.
            shape = [4]
            dtype = "int8"
            self.assertRaises(TypeError, fluid.layers.zeros, shape, dtype)


class ApiZerosTest(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            zeros = paddle.zeros(shape=[10], dtype="float64")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            zeros = paddle.zeros(shape=[10], dtype="int64")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            zeros = paddle.zeros(shape=[10], dtype="int64")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)


class ApiZerosError(unittest.TestCase):
    def test_errors(self):
        def test_error1():
            with fluid.program_guard(fluid.Program()):
                ones = fluid.layers.zeros(shape=10, dtype="int64")

        self.assertRaises(TypeError, test_error1)

        def test_error2():
            with fluid.program_guard(fluid.Program()):
                ones = fluid.layers.zeros(shape=[10], dtype="int8")

        self.assertRaises(TypeError, test_error2)


if __name__ == "__main__":
    unittest.main()
