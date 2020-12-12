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

import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci


def nearest_neighbor_interp_mkldnn_np(X,
                                      out_h,
                                      out_w,
                                      out_size=None,
                                      data_layout='NCHW'):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        X = np.transpose(X, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]

    n, c, in_h, in_w = X.shape

    fh = fw = 0.0
    if (out_h > 1):
        fh = out_h * 1.0 / in_h
    if (out_w > 1):
        fw = out_w * 1.0 / in_w

    out = np.zeros((n, c, out_h, out_w))

    for oh in range(out_h):
        ih = int(round((oh + 0.5) / fh - 0.5))
        for ow in range(out_w):
            iw = int(round((ow + 0.5) / fw - 0.5))
            # print(ih, iw)
            out[:, :, oh, ow] = X[:, :, ih, iw]

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(X.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestNearestInterpMKLDNNOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "nearest_interp"
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {
            'interp_method': self.interp_method,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_layout
        }

        input_np = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float32")
        elif self.scale > 0:
            out_h = int(self.input_shape[2] * self.scale)
            out_w = int(self.input_shape[3] * self.scale)
            self.attrs['scale'] = self.scale
        else:
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs['SizeTensor'] = size_tensor
            self.inputs['OutSize'] = self.out_size

        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
        output_np = nearest_neighbor_interp_mkldnn_np(input_np, out_h, out_w,
                                                      self.out_size)
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 2, 2]
        self.out_h = 4
        self.out_w = 4
        self.scale = 2.0
        self.out_size = None
        self.scale_by_1Dtensor = True
        self.use_mkldnn = True
        self.data_layout = 'NCHW'


class TestNearestInterpOpMKLDNNNHWC(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 27
        self.out_w = 49
        self.scale = 2.0
        self.out_size = None
        self.scale_by_1Dtensor = True
        self.use_mkldnn = True
        self.data_layout = 'NHWC'


class TestNearestNeighborInterpMKLDNNCase2(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 1.
        self.use_mkldnn = True


class TestNearestNeighborInterpDataLayout(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 4, 4, 5]
        self.out_h = 6
        self.out_w = 7
        self.scale = 0.
        self.data_layout = "NHWC"
        self.use_mkldnn = True


class TestNearestNeighborInterpCase3(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 128
        self.scale = 0.
        self.use_mkldnn = True


class TestNearestNeighborInterpCase4(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")


# if out_h=1 and out_w=1, oneDNN downsampling will use the center value
# any downsampling can not pass because of formula difference
# self.out_size = np.array([13, 13]).astype("int32") can not pass because of formula difference
class TestNearestNeighborInterpCase5(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([13, 13]).astype("int32")
        self.use_mkldnn = True


class TestNearestNeighborInterpCase6(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 129]).astype("int32")
        self.use_mkldnn = True


class TestNearestNeighborInterpSame(TestNearestInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.use_mkldnn = True


if __name__ == "__main__":
    unittest.main()
