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
from paddle.fluid import core
from paddle.fluid import Program, program_guard


class TestDiagFlatError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):

            def test_diagflat_type():
                x = [1, 2, 3]
                output = paddle.diagflat(x)

            self.assertRaises(TypeError, test_diagflat_type)

            x = paddle.static.data('data', [3, 3])
            self.assertRaises(TypeError, paddle.diagflat, x, offset=2.5)

            self.assertRaises(TypeError, paddle.diagflat, x, padding_value=[9])


class TestDiagFlatAPI(unittest.TestCase):
    def setUp(self):
        self.input_np = np.random.random(size=(10, 10)).astype(np.float32)
        self.expected0 = np.diagflat(self.input_np)
        self.expected1 = np.diagflat(self.input_np, k=1)
        self.expected2 = np.diagflat(self.input_np, k=-1)

        self.input_np2 = np.random.rand(100)
        self.offset = 0
        self.padding_value = 8
        n = self.input_np2.size
        self.expected3 = self.padding_value * np.ones(
            (n, n)) + np.diagflat(self.input_np2, self.offset) - np.diagflat(
                self.padding_value * np.ones(n))

        self.input_np3 = np.random.randint(-10, 10, size=(100)).astype(np.int64)
        self.padding_value = 8.0
        n = self.input_np3.size
        self.expected4 = self.padding_value * np.ones(
            (n, n)) + np.diagflat(self.input_np3, self.offset) - np.diagflat(
                self.padding_value * np.ones(n))

        self.padding_value = -8
        self.expected5 = self.padding_value * np.ones(
            (n, n)) + np.diagflat(self.input_np3, self.offset) - np.diagflat(
                self.padding_value * np.ones(n))

        self.input_np4 = np.random.random(size=(100, 100)).astype(np.float32)
        self.expected6 = np.diagflat(self.input_np4)
        self.expected7 = np.diagflat(self.input_np4, k=1)
        self.expected8 = np.diagflat(self.input_np4, k=-1)

        self.input_np5 = np.random.random(size=(2000)).astype(np.float32)
        self.expected9 = np.diagflat(self.input_np5)
        self.expected10 = np.diagflat(self.input_np5, k=1)
        self.expected11 = np.diagflat(self.input_np5, k=-1)

        self.input_np6 = np.random.random(size=(200, 50)).astype(np.float32)
        self.expected12 = np.diagflat(self.input_np6, k=-1)

    def run_imperative(self):
        x = paddle.to_tensor(self.input_np)
        y = paddle.diagflat(x)
        self.assertTrue(np.allclose(y.numpy(), self.expected0))

        y = paddle.diagflat(x, offset=1)
        self.assertTrue(np.allclose(y.numpy(), self.expected1))

        y = paddle.diagflat(x, offset=-1)
        self.assertTrue(np.allclose(y.numpy(), self.expected2))

        x = paddle.to_tensor(self.input_np2)
        y = paddle.diagflat(x, padding_value=8)
        self.assertTrue(np.allclose(y.numpy(), self.expected3))

        x = paddle.to_tensor(self.input_np3)
        y = paddle.diagflat(x, padding_value=8.0)
        self.assertTrue(np.allclose(y.numpy(), self.expected4))

        y = paddle.diagflat(x, padding_value=-8)
        self.assertTrue(np.allclose(y.numpy(), self.expected5))

        x = paddle.to_tensor(self.input_np4)
        y = paddle.diagflat(x)
        self.assertTrue(np.allclose(y.numpy(), self.expected6))

        y = paddle.diagflat(x, offset=1)
        self.assertTrue(np.allclose(y.numpy(), self.expected7))

        y = paddle.diagflat(x, offset=-1)
        self.assertTrue(np.allclose(y.numpy(), self.expected8))

        x = paddle.to_tensor(self.input_np5)
        y = paddle.diagflat(x)
        self.assertTrue(np.allclose(y.numpy(), self.expected9))

        y = paddle.diagflat(x, offset=1)
        self.assertTrue(np.allclose(y.numpy(), self.expected10))

        y = paddle.diagflat(x, offset=-1)
        self.assertTrue(np.allclose(y.numpy(), self.expected11))

        x = paddle.to_tensor(self.input_np6)
        y = paddle.diagflat(x, offset=-1)
        self.assertTrue(np.allclose(y.numpy(), self.expected12))

    def run_static(self, use_gpu=False):
        x = paddle.static.data(name='input', shape=[10, 10], dtype='float32')
        x2 = paddle.static.data(name='input2', shape=[100], dtype='float64')
        x3 = paddle.static.data(name='input3', shape=[100], dtype='int64')
        x4 = paddle.static.data(
            name='input4', shape=[100, 100], dtype='float32')
        x5 = paddle.static.data(name='input5', shape=[2000], dtype='float32')
        x6 = paddle.static.data(name='input6', shape=[200, 50], dtype='float32')
        result0 = paddle.diagflat(x)
        result1 = paddle.diagflat(x, offset=1)
        result2 = paddle.diagflat(x, offset=-1)
        result3 = paddle.diagflat(x, name='aaa')
        result4 = paddle.diagflat(x2, padding_value=8)
        result5 = paddle.diagflat(x3, padding_value=8.0)
        result6 = paddle.diagflat(x3, padding_value=-8)
        result7 = paddle.diagflat(x4)
        result8 = paddle.diagflat(x4, offset=1)
        result9 = paddle.diagflat(x4, offset=-1)
        result10 = paddle.diagflat(x5)
        result11 = paddle.diagflat(x5, offset=1)
        result12 = paddle.diagflat(x5, offset=-1)
        result13 = paddle.diagflat(x6, offset=-1)

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        res0, res1, res2, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13 = exe.run(
            feed={
                "input": self.input_np,
                "input2": self.input_np2,
                'input3': self.input_np3,
                'input4': self.input_np4,
                'input5': self.input_np5,
                'input6': self.input_np6
            },
            fetch_list=[
                result0, result1, result2, result4, result5, result6, result7,
                result8, result9, result10, result11, result12, result13
            ])

        self.assertTrue(np.allclose(res0, self.expected0))
        self.assertTrue(np.allclose(res1, self.expected1))
        self.assertTrue(np.allclose(res2, self.expected2))
        self.assertTrue('aaa' in result3.name)
        self.assertTrue(np.allclose(res4, self.expected3))
        self.assertTrue(np.allclose(res5, self.expected4))
        self.assertTrue(np.allclose(res6, self.expected5))
        self.assertTrue(np.allclose(res7, self.expected6))
        self.assertTrue(np.allclose(res8, self.expected7))
        self.assertTrue(np.allclose(res9, self.expected8))
        self.assertTrue(np.allclose(res10, self.expected9))
        self.assertTrue(np.allclose(res11, self.expected10))
        self.assertTrue(np.allclose(res12, self.expected11))
        self.assertTrue(np.allclose(res13, self.expected12))

    def test_cpu(self):
        paddle.disable_static(place=paddle.fluid.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.fluid.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        with fluid.program_guard(fluid.Program()):
            self.run_static(use_gpu=True)


if __name__ == "__main__":
    unittest.main()
