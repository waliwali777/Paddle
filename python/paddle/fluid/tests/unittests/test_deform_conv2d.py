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

import paddle
import paddle.nn.functional as F
import paddle.nn.initializer as I
import numpy as np
import unittest
from unittest import TestCase

class TestDeformConv2D(TestCase):
    batch_size = 1
    spatial_shape = (3, 3)
    dtype = "float32"

    def setUp(self):
        self.in_channels = 1
        self.out_channels = 3
        self.kernel_size = [3, 3]
        self.padding = [1, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False

    def prepare(self):
        np.random.seed(1)
        paddle.seed(1)
        if isinstance(self.kernel_size, int):
            filter_shape = (self.kernel_size, ) * 2
        else:
            filter_shape = tuple(self.kernel_size)
        self.filter_shape = filter_shape

        self.weight = np.random.uniform(
            -1, 1, (self.out_channels, self.in_channels // self.groups
                    ) + filter_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (
                self.out_channels, )).astype(self.dtype)

        def out_size(in_size, pad_size, dilation_size, kernel_size,
                     stride_size):
            return (in_size + 2 * pad_size -
                    (dilation_size * (kernel_size - 1) + 1)) / stride_size + 1

        out_h = int(
            out_size(self.spatial_shape[0], self.padding[0], self.dilation[0],
                     self.kernel_size[0], self.stride[0]))
        out_w = int(
            out_size(self.spatial_shape[1], self.padding[1], self.dilation[1],
                     self.kernel_size[1], self.stride[1]))
        out_shape = (out_h, out_w)

        self.input_shape = (self.batch_size, self.in_channels
                            ) + self.spatial_shape

        self.offset_shape = (self.batch_size, self.deformable_groups * 2 *
                             filter_shape[0] * filter_shape[1]) + out_shape

        self.mask_shape = (self.batch_size, self.deformable_groups *
                           filter_shape[0] * filter_shape[1]) + out_shape

        self.input = np.random.uniform(-1, 1,
                                       self.input_shape).astype(self.dtype)

        self.offset = np.random.uniform(-1, 1,
                                        self.offset_shape).astype(self.dtype)

        self.mask = np.random.uniform(-1, 1, self.mask_shape).astype(self.dtype)

    def static_graph_case_dcn(self):
        main = paddle.static.Program()
        start = paddle.static.Program()
        paddle.enable_static()
        with paddle.static.program_guard(main, start):
            x = paddle.static.data(
                "input", (-1, self.in_channels, -1, -1), dtype=self.dtype)
            offset = paddle.static.data(
                "offset", (-1, self.deformable_groups * 2 *
                           self.filter_shape[0] * self.filter_shape[1], -1, -1),
                dtype=self.dtype)
            mask = paddle.static.data(
                "mask", (-1, self.deformable_groups * self.filter_shape[0] *
                         self.filter_shape[1], -1, -1),
                dtype=self.dtype)

            y_v1 = paddle.fluid.layers.deformable_conv(
                input=x,
                offset=offset,
                mask=None,
                num_filters=self.out_channels,
                filter_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                im2col_step=1,
                param_attr=I.Assign(self.weight),
                bias_attr=False if self.no_bias else I.Assign(self.bias),
                modulated=False)

        exe = paddle.static.Executor(self.place)
        exe.run(start)
        out_v1 = exe.run(main,feed={
                                    "input": self.input,
                                    "offset": self.offset,
                                    "mask": self.mask },
                                fetch_list=[y_v1])
        return out_v1

    def dygraph_case_dcn(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.input)
        offset = paddle.to_tensor(self.offset)
        mask = paddle.to_tensor(self.mask)
        bias = None if self.no_bias else paddle.to_tensor(self.bias)

        deform_conv2d = paddle.vision.ops.DeformConv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            deformable_groups=self.deformable_groups,
            groups=self.groups,
            weight_attr=I.Assign(self.weight),
            bias_attr=False if self.no_bias else I.Assign(self.bias))

        y_v1 = deform_conv2d(x, offset)
        out_v1 = y_v1.numpy()

        return out_v1

    def _test_identity(self):
        self.prepare()
        static_dcn_v1 = self.static_graph_case_dcn()
        dy_dcn_v1 = self.dygraph_case_dcn()
        
        print(">> static_dcn_v1 : \n", static_dcn_v1)
        print(">> dy_dcn_v1 : \n", dy_dcn_v1)

        np.testing.assert_array_almost_equal(static_dcn_v1, dy_dcn_v1)

    def test_identity(self):
        self.place = paddle.CPUPlace()
        self._test_identity()

if __name__ == "__main__":
    unittest.main()
