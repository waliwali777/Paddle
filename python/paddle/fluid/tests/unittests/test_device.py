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
from op_test import OpTest

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import warnings
import paddle


class TestStaticDeviceManage(unittest.TestCase):
    def test_cpu_device(self):
        paddle.set_device('cpu')
        out1 = paddle.zeros(shape=[1, 3], dtype='float32')
        out2 = paddle.ones(shape=[1, 3], dtype='float32')
        out3 = paddle.concat(x=[out1, out2], axis=0)
        exe = paddle.fluid.Executor()
        exe.run(paddle.fluid.default_startup_program())
        res = exe.run(fetch_list=[out3])
        self.assertEqual(isinstance(exe.place, core.CPUPlace), True)

    def test_gpu_device(self):
        if core.is_compiled_with_cuda():
            out1 = paddle.zeros(shape=[1, 3], dtype='float32')
            out2 = paddle.ones(shape=[1, 3], dtype='float32')
            out3 = paddle.concat(x=[out1, out2], axis=0)
            paddle.set_device('gpu:0')
            exe = paddle.fluid.Executor()
            exe.run(paddle.fluid.default_startup_program())
            res = exe.run(fetch_list=[out3])
            self.assertEqual(isinstance(exe.place, core.CUDAPlace), True)
            device_id = core.get_device_id()
            self.assertEqual(device_id, 0)


class TestImperativeDeviceManage(unittest.TestCase):
    def test_cpu(self):
        with fluid.dygraph.guard():
            paddle.set_device('cpu')
            out1 = paddle.zeros(shape=[1, 3], dtype='float32')
            out2 = paddle.ones(shape=[1, 3], dtype='float32')
            out3 = paddle.concat(x=[out1, out2], axis=0)
            self.assertEqual(
                isinstance(framework._current_expected_place(), core.CPUPlace),
                True)

    def test_gpu(self):
        with fluid.dygraph.guard():
            paddle.set_device('gpu:0')
            out1 = paddle.zeros(shape=[1, 3], dtype='float32')
            out2 = paddle.ones(shape=[1, 3], dtype='float32')
            out3 = paddle.concat(x=[out1, out2], axis=0)
            self.assertEqual(
                isinstance(framework._current_expected_place(), core.CUDAPlace),
                True)
            device_id = core.get_device_id()
            self.assertEqual(device_id, 0)


if __name__ == '__main__':
    unittest.main()
