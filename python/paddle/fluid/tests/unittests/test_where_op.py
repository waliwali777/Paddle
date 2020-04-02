# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.tensor as tensor
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.op import Operator
from paddle.fluid.backward import append_backward


class TestWhereOp(OpTest):
    def setUp(self):
        self.op_type = "where"
        self.init_config()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def init_config(self):
        self.x = np.random.uniform(-3, 5, (100)).astype("float64")
        self.y = np.random.uniform(-3, 5, (100)).astype("float64")
        self.cond = np.zeros((100)).astype("bool")

        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}


class TestWhereOp2(TestWhereOp):
    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def init_config(self):
        self.x = np.random.uniform(-5, 5, (60, 2)).astype("float64")
        self.y = np.random.uniform(-5, 5, (60, 2)).astype("float64")
        self.cond = np.ones((60, 2)).astype("bool")

        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}


class TestWhereOp3(TestWhereOp):
    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def init_config(self):
        self.x = np.random.uniform(-3, 5, (20, 2, 4)).astype("float64")
        self.y = np.random.uniform(-3, 5, (20, 2, 4)).astype("float64")
        self.cond = np.array(np.random.randint(2, size=(20, 2, 4)), dtype=bool)

        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}


class TestWhereAPI(unittest.TestCase):
    def test_api(self, use_cuda=False):
        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[4], dtype='float32')
            y = fluid.layers.data(name='y', shape=[4], dtype='float32')
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float32")
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float32")
            cond_i = np.array([False, False, True, True]).astype("bool")
            result = tensor.where(x > 1, X=x, Y=y)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(fluid.default_main_program(),
                          feed={'x': x_i,
                                'y': y_i},
                          fetch_list=[result])
            assert np.array_equal(out[0], np.where(cond_i, x_i, y_i))

    def test_grad(self, use_cuda=False):
        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[4], dtype='float32')
            y = fluid.layers.data(name='y', shape=[4], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float32")
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float32")
            cond_i = np.array([False, False, True, True]).astype("bool")
            result = tensor.where(x > 1, X=x, Y=y)
            x_mean = layers.mean(x)
            append_backward(x_mean)
            y_mean = layers.mean(y)
            append_backward(y_mean)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(fluid.default_main_program(),
                          feed={'x': x_i,
                                'y': y_i},
                          fetch_list=[result, x.grad_name, y.grad_name])
            x_grad = [0.25] * 4
            y_grad = [0.25] * 4
            assert np.array_equal(out[0], np.where(cond_i, x_i, y_i))
            assert np.array_equal(out[1], x_grad)
            assert np.array_equal(out[2], y_grad)

    def test_grad1(self, use_cuda=False):
        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[4], dtype='float32')
            y = fluid.layers.data(name='y', shape=[4], dtype='float32')
            x.stop_gradient = True
            y.stop_gradient = False
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float32")
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float32")
            cond_i = np.array([False, False, True, True]).astype("bool")
            result = tensor.where(x > 1, X=x, Y=y)
            y_mean = layers.mean(y)
            append_backward(y_mean)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(fluid.default_main_program(),
                          feed={'x': x_i,
                                'y': y_i},
                          fetch_list=[result, y.grad_name])
            y_grad = [0.25] * 4
            assert np.array_equal(out[0], np.where(cond_i, x_i, y_i))
            assert np.array_equal(out[1], y_grad)

    def test_grad2(self, use_cuda=False):
        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[4], dtype='float32')
            y = fluid.layers.data(name='y', shape=[4], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = True
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float32")
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float32")
            cond_i = np.array([False, False, True, True]).astype("bool")
            result = tensor.where(x > 1, X=x, Y=y)
            x_mean = layers.mean(x)
            append_backward(x_mean)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(fluid.default_main_program(),
                          feed={'x': x_i,
                                'y': y_i},
                          fetch_list=[result, x.grad_name])
            x_grad = [0.25] * 4
            assert np.array_equal(out[0], np.where(cond_i, x_i, y_i))
            assert np.array_equal(out[1], x_grad)

    def test_fw_bw(self):
        if core.is_compiled_with_cuda():
            self.test_api(use_cuda=True)
            self.test_grad(use_cuda=True)
            self.test_grad1(use_cuda=True)
            self.test_grad2(use_cuda=True)
        self.test_api(use_cuda=False)
        self.test_grad(use_cuda=False)
        self.test_grad1(use_cuda=False)
        self.test_grad2(use_cuda=False)


class TestWhereDygraphAPI(unittest.TestCase):
    def test_api(self):
        x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float64")
        y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float64")
        cond_i = np.array([False, False, True, True]).astype("bool")

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(x_i)
            y = fluid.dygraph.to_variable(y_i)
            cond = fluid.dygraph.to_variable(cond_i)
            out = tensor.where(cond, x, y)
            assert np.array_equal(out.numpy(), np.where(cond_i, x_i, y_i))


if __name__ == '__main__':
    unittest.main()
