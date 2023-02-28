# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from utils import TOLERANCE

np.random.seed(2013)

import paddle
import paddle.nn.functional as F
from paddle.fluid import core


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = "float32"
        self.shape = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype
        return

    def set_shape(self, shape) -> None:
        self.shape = shape
        return

    def get_rtol(self, flag):
        rtol = TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x):
    return F.hardswish(x)


def expect_forward(inputs):
    return fn(inputs)


class TestCompositeHardswish(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float16", "float32", "float64"]
        self.shapes = [[16, 16, 64, 64], [2, 3, 4], [2, 3]]

    def cal_composite(self, inputs):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = fn(x)
            blocks = main_program.blocks
            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that hardswish in original block
            self.assertTrue('hard_swish' in fwd_ops)
            paddle.incubate.autograd.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that hardswish is splitted into small ops
            self.assertTrue('hard_swish' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        return res

    def compare_forward(self):
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)

        expect = expect_forward(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        for j in self.dtypes:
            for t in self.shapes:
                # hardswish-kernel on cpu not support float16
                if paddle.device.get_device() == "cpu" and j == "float16":
                    print("need pass this case")
                    continue
                attrs.set_dtype(j)
                attrs.set_shape(t)
                self.compare_forward()


if __name__ == '__main__':
    unittest.main()
