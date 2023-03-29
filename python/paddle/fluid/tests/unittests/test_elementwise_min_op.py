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

import unittest

import numpy as np
from eager_op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
from paddle.fluid import core

paddle.enable_static()


def broadcast_wrapper(shape=[1, 10, 12, 1]):
    def min_wrapper(x, y, axis=-1):
        return paddle.minimum(x, y.reshape(shape))

    return min_wrapper


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X")
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y')
        )


class TestElementwiseFP16Op(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float16)
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output(atol=1e-3)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=1e-3)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=1e-3, no_grad_set=set("X")
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=1e-3, no_grad_set=set('Y')
        )


class TestElementwiseMinOp_ZeroDim1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.1, 1, []).astype("float64")
        y = np.random.uniform(0.1, 1, []).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_ZeroDim1(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype(np.float16)
        self.y = np.random.uniform(0.1, 1, []).astype(np.float16)


class TestElementwiseMinOp_ZeroDim2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        y = np.random.uniform(0.1, 1, []).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_ZeroDim2(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype("float16")
        self.y = np.random.uniform(0.1, 1, []).astype("float16")


class TestElementwiseMinOp_ZeroDim3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.1, 1, []).astype("float64")
        y = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_ZeroDim3(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float16")
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype("float16")


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMinOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.random_integers(-5, 5, [10, 3, 4]).astype("float64")
        y = np.array([0.5]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMinFP16Op_scalar(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.random_integers(-5, 5, [10, 3, 4]).astype(np.float16)
        y = np.array([0.5]).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.random((100,)).astype("float64")
        sgn = np.random.choice([-1, 1], (100,)).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, (100,)).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_Vector(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.random((100,)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x + sgn * np.random.uniform(0.1, 1, (100,)).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[100, 1, 1])
        x = np.random.uniform(0.5, 1, (100, 3, 2)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[:, 0, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1)
            )
        }


class TestElementwiseMinFP16Op_broadcast_0(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[100, 1, 1])
        x = np.random.uniform(0.5, 1, (100, 3, 2)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x[:, 0, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float16
        )
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1)
            )
        }


class TestElementwiseMinOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 100, 1])
        x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[0, :, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1)
            )
        }


class TestElementwiseMinFP16Op_broadcast_1(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 100, 1])
        x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x[0, :, 0] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float16
        )
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1)
            )
        }


class TestElementwiseMinOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 1, 100])
        x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
        }


class TestElementwiseMinFP16Op_broadcast_2(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 1, 100])
        x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float16
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
        }


class TestElementwiseMinOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 25, 4, 1])
        x = np.random.uniform(0.5, 1, (2, 25, 4, 1)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (25, 4)).astype(np.float64)
        y = x[0, :, :, 0] + sgn * np.random.uniform(1, 2, (25, 4)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 25, 4, 1)
            )
        }


class TestElementwiseMinFP16Op_broadcast_3(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 25, 4, 1])
        x = np.random.uniform(0.5, 1, (2, 25, 4, 1)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (25, 4)).astype(np.float16)
        y = x[0, :, :, 0] + sgn * np.random.uniform(1, 2, (25, 4)).astype(
            np.float16
        )
        self.inputs = {'X': x, 'Y': y}

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 25, 4, 1)
            )
        }


class TestElementwiseMinOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(np.float64)
        y = x + sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_broadcast_4(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(np.float16)
        y = x + sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


@unittest.skipIf(
    core.is_compiled_with_cuda()
    and (
        core.cudnn_version() < 8100
        or paddle.device.cuda.get_device_capability()[0] < 8
    ),
    "run test when gpu is availble and the minimum cudnn version is 8.1.0 and gpu's compute capability is at least 8.0.",
)
class TestElementwiseBF16Op(OpTest):
    def init_data(self):
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float32)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            np.float32
        )

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.enable_cinn = False
        self.dtype = np.uint16
        self.inputs = {
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(np.minimum(self.x, self.y))
        }

    def test_check_output(self):
        self.check_output(atol=1e-2)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', numeric_grad_delta=0.05)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', numeric_grad_delta=0.05, no_grad_set=set("X")
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', numeric_grad_delta=0.05, no_grad_set=set('Y')
        )


class TestElementwiseMinBF16Op_ZeroDim1(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")


class TestElementwiseMinBF16Op_scalar(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype("float32")
        self.y = np.array([0.5]).astype("float32")
        self.__class__.no_need_check_grad = True


class TestElementwiseMinBF16Op_Vector(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.random((100,)).astype("float32")
        sgn = np.random.choice([-1, 1], (100,)).astype("float32")
        self.y = self.x + sgn * np.random.uniform(0.1, 1, (100,)).astype(
            "float32"
        )


if __name__ == '__main__':
    unittest.main()
