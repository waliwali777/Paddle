#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce
from operator import mul

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def ref_rms_norm(
    x,
    norm_weight,
    norm_bias=None,
    epsilon=1e-5,
    begin_norm_axis=1,
    residual=None,
    bias=None,
):
    # print('x_input', x)
    x_shape = x.shape
    left = reduce(mul, x_shape[0:begin_norm_axis], 1)
    right = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [left, right]
    if residual is not None:
        x = x + residual.reshape(x.shape)
    if bias is not None:
        x = x + bias.reshape([1, right])
    variance = np.pow(x, 2).mean(-1, keepdims=True) + epsilon
    y = np.sqrt(variance).reshape([left, 1])

    y = norm_weight.reshape([1, right]) * y
    if norm_bias is not None:
        y = y + norm_bias.reshape([1, right])
    y.shape = x.shape
    variance.shape = [left, 1]
    variance = variance.astype('float')
    # print('x_output', x)
    # print(norm_weight)

    if residual is not None:
        return y, x, variance
    else:
        return y, None, variance


class XPUTestRmsNormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'rms_norm'
        self.use_dynamic_create_class = False

    class TestXPURmsNormOp(XPUOpTest):
        def setUp(self):
            self.__class__.use_xpu = True
            self.set_no_need_check_grad()
            self.op_type = "rms_norm"
            if self.in_type == np.uint16:
                self.dtype = np.float32
            else:
                self.dtype = self.in_type
            self.shape = [2, 3, 4, 5]
            self.epsilon = 1e-05
            self.begin_norm_axis = 1
            self.use_norm_bias = False  # optional input
            self.use_residual = False  # optional input
            self.use_bias = False  # optional input

            self.set_attrs()

            self.atol = 1e-4
            if self.in_type == np.float16 or self.in_type == np.uint16:
                self.atol = 1e-2

            left = reduce(mul, self.shape[0 : self.begin_norm_axis], 1)
            right = reduce(
                mul, self.shape[self.begin_norm_axis : len(self.shape)], 1
            )
            np.random.seed(10)

            x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            norm_weight_np = np.random.uniform(-1, 1, [right]).astype(
                self.dtype
            )
            if self.use_norm_bias:
                norm_bias_np = np.random.uniform(-1, 1, [right]).astype(
                    self.dtype
                )
            else:
                norm_bias_np = None
            if self.use_residual:
                residual_np = np.random.uniform(-1, 1, self.shape).astype(
                    self.dtype
                )
            else:
                residual_np = None
            if self.use_bias:
                bias_np = np.random.uniform(-1, 1, [right]).astype(self.dtype)
            else:
                bias_np = None

            # CPU

            ref_y_np, ref_residual_out, ref_variance_np = ref_rms_norm(
                x_np,
                norm_weight_np,
                norm_bias_np,
                self.epsilon,
                self.begin_norm_axis,
                residual_np,
                bias_np,
            )
            # print('dtype', self.in_type)
            if self.in_type == np.uint16:
                ref_y_np = convert_float_to_uint16(ref_y_np)
                if self.use_residual:
                    ref_residual_out = convert_float_to_uint16(ref_residual_out)

            if self.in_type == np.uint16:  # bfloat16 actually
                x_np = convert_float_to_uint16(x_np)
                norm_weight_np = convert_float_to_uint16(norm_weight_np)
                if self.use_norm_bias:
                    norm_bias_np = convert_float_to_uint16(norm_bias_np)
                if self.use_residual:
                    residual_np = convert_float_to_uint16(residual_np)
                if self.use_bias:
                    bias_np = convert_float_to_uint16(bias_np)
            print('######################')
            print('dtype', self.in_type)
            print('shape', self.shape)
            print('begin_norm_axis', self.begin_norm_axis)
            self.inputs = {
                'X': x_np,
                'Norm_weight': norm_weight_np,
            }

            if self.use_norm_bias:
                self.inputs['Norm_bias'] = norm_bias_np
            if self.use_residual:
                self.inputs['Residual'] = residual_np
            if self.use_bias:
                self.inputs['Bias'] = bias_np

            self.attrs = {
                'epsilon': self.epsilon,
                'begin_norm_axis': self.begin_norm_axis,
                'quant_scale': -1,
                'quant_round_type': 0,
                'quant_max_bound': 0,
                'quant_min_bound': 0,
            }
            # print('shape', self.shape)
            # print('inputs', self.inputs)
            # print('attrs', self.attrs)
            # print('x', x_np.shape)
            # print('norm_weight_np', norm_weight_np.shape)
            if self.use_residual:
                self.outputs = {
                    'Y': ref_y_np,
                    # 'Residual_out': ref_residual_out,
                    'Inv_var': ref_variance_np,
                }
            else:
                self.outputs = {'Y': ref_y_np, 'Inv_var': ref_variance_np}

        def set_no_need_check_grad(self):
            self.__class__.no_need_check_grad = False

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0), atol=self.atol)

        def test_check_grad(self):
            self.check_grad_with_place(
                paddle.XPUPlace(0), ['X'], 'Y', max_relative_error=self.atol
            )

    class TestXPURmsNormOpAxis2(TestXPURmsNormOp):
        def set_attrs(self):
            self.begin_norm_axis = 2

    # class TestXPURmsNormOpAxis3(TestXPURmsNormOp):
    #     def set_attrs(self):
    #         self.begin_norm_axis = 3

    # class TestXPURmsNormOp2D(TestXPURmsNormOp):
    #     def set_attrs(self):
    #         self.shape = [10, 12]

    # class TestXPURmsNormOp3D(TestXPURmsNormOp):
    #     def set_attrs(self):
    #         self.shape = [4, 5, 6]

    # class TestXPURmsNormOpNormBias(TestXPURmsNormOp):
    #     def set_attrs(self):
    #         self.use_norm_bias = True

    # class TestXPURmsNormOpResidual1(TestXPURmsNormOp):
    #     def set_no_need_check_grad(self):
    #         self.__class__.no_need_check_grad = True
    #     def set_attrs(self):
    #         self.use_residual = True

    # class TestXPURmsNormOpResidual2(TestXPURmsNormOp):
    #     def set_no_need_check_grad(self):
    #         self.__class__.no_need_check_grad = True
    #     def set_attrs(self):
    #         self.use_residual = True
    #         self.use_bias = True

    # class TestXPURmsNormOpResidual3(TestXPURmsNormOp):
    #     def set_no_need_check_grad(self):
    #         self.__class__.no_need_check_grad = True
    #     def set_attrs(self):
    #         self.use_residual = True
    #         self.use_bias = True
    #         self.use_norm_bias = True


support_types = get_xpu_op_support_types('rms_norm')

for stype in support_types:
    print(stype)
    if stype == 'float32':
        create_test_class(globals(), XPUTestRmsNormOp, stype)


if __name__ == "__main__":
    unittest.main()
