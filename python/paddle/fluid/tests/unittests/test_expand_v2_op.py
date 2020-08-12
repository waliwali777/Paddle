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
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle


# Situation 1: shape is a list(without tensor)
class TestExpandV2OpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()

        self.inputs = {'X': np.random.random(self.ori_shape).astype("float64")}
        self.attrs = {'shape': self.shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.shape = [200]
        self.expand_times = [2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandV2OpRank2_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.shape = [240]
        self.expand_times = [2]


class TestExpandV2OpRank2(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.shape = [24, 42]
        self.expand_times = [2, 3]


class TestExpandV2OpRank3_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandV2OpRank3(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 15)
        self.shape = (4, 4, 60)
        self.expand_times = (2, 1, 4)


class TestExpandV2OpRank4(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.shape = (6, 8, 5, 14)
        self.expand_times = (3, 2, 1, 2)


# Situation 2: shape is a list(with tensor)
class TestExpandV2OpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()
        expand_shapes_tensor = []
        for index, ele in enumerate(self.expand_shape):
            expand_shapes_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype("float64"),
            'expand_shapes_tensor': expand_shapes_tensor,
        }
        self.attrs = {"shape": self.infer_expand_shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2]
        self.expand_shape = [200]
        self.infer_expand_shape = [-1]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandV2OpRank2_Corner_tensor_attr(TestExpandV2OpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.expand_shape = [12, 14]
        self.infer_expand_shape = [12, -1]


class TestExpandV2OpRank2_attr_tensor(TestExpandV2OpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]
        self.expand_shape = [24, 42]
        self.infer_expand_shape = [-1, 42]


# Situation 3: shape is a tensor
class TestExpandV2OpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.init_data()

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype("float64"),
            'Shape': np.array(self.expand_shape).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2]
        self.expand_shape = [200]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandV2OpRank2_tensor(TestExpandV2OpRank1_tensor):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]
        self.expand_shape = [24, 42]


# Situation 4: input x is Integer
class TestExpandV2OpInteger(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int32")
        }
        self.attrs = {'shape': [4, 4, 20]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 5: input x is Bool
class TestExpandV2OpBoolean(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {'shape': [4, 4, 20]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


# Situation 56: input x is Integer
class TestExpandV2OpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.inputs = {
            'X': np.random.randint(
                10, size=(2, 4, 5)).astype("int64")
        }
        self.attrs = {'shape': [4, 4, 20]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestExpandV2Error(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            shape = [2, 2]
            self.assertRaises(TypeError, paddle.tensor.expand, x1, shape)
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tensor.expand, x2, shape)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="bool")
            x3.stop_gradient = True
            self.assertRaises(ValueError, paddle.tensor.expand, x3, shape)


# Test python API
class TestExpandV2API(unittest.TestCase):
    def test_api(self):
        input = np.random.random([12, 14]).astype("float32")
        x = fluid.layers.data(
            name='x', shape=[12, 14], append_batch_size=False, dtype="float32")

        positive_2 = fluid.layers.fill_constant([1], "int32", 24)
        expand_shape = fluid.layers.data(
            name="expand_shape", shape=[2], append_batch_size=False)

        out_1 = paddle.tensor.expand(x, shape=[24, 42])
        out_2 = paddle.tensor.expand(x, shape=[positive_2, 42])
        out_3 = paddle.tensor.expand(x, shape=expand_shape)

        g0 = fluid.backward.calc_gradient(out_2, x)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3 = exe.run(fluid.default_main_program(),
                                      feed={
                                          "x": input,
                                          "expand_shape":
                                          np.array([12, 42]).astype("int32")
                                      },
                                      fetch_list=[out_1, out_2, out_3])
        assert np.array_equal(res_1, np.tile(input, (2, 3)))
        assert np.array_equal(res_2, np.tile(input, (2, 3)))
        assert np.array_equal(res_3, np.tile(input, (1, 3)))


if __name__ == "__main__":
    unittest.main()
