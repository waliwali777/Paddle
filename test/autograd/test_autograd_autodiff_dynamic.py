# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import typing
import unittest

import config
import numpy as np
import utils

import paddle
import paddle.nn.functional as F
from paddle.incubate.autograd.utils import as_tensors


def make_v(f, inputs):
    outputs = as_tensors(f(*inputs))
    return [paddle.ones_like(x) for x in outputs]


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'func', 'xs'),
    (
        ('1d_in_1d_out', utils.square, np.array([2.0, 3.0])),
        ('3d_in_3d_out', utils.square, np.random.rand(2, 3, 4)),
        ('single_in_single_out', utils.square, np.random.rand(2, 3)),
        (
            'multi_in_single_out',
            paddle.matmul,
            (np.random.rand(2, 2), np.random.rand(2, 2)),
        ),
    ),
)
class TestJacobianNoBatch(unittest.TestCase):
    def setUp(self):
        self._dtype = (
            self.xs[0].dtype
            if isinstance(self.xs, typing.Sequence)
            else self.xs.dtype
        )
        self._eps = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("eps")
        )
        self._rtol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        self._atol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_jacobian(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=None)
        if isinstance(self._actual, (tuple, list)):
            self._actual = paddle.concat([x[:] for x in self._actual], axis=1)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index('all', (slice(0, None, None), slice(0, None, None))),
            Index('row', (0, slice(0, None, None))),
            Index('col', (slice(0, None, None), 0)),
            Index('multi-row', (slice(0, 2, 1), slice(0, None, None))),
        )
        self.assertEqual(self._actual[:].numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def test_jacobian_attribute_operator(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=None)
        if isinstance(self._actual, (tuple, list)):
            self._actual = paddle.concat([x[:] for x in self._actual], axis=1)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index('all', (slice(0, None, None), slice(0, None, None))),
            Index('row', (0, slice(0, None, None))),
            Index('col', (slice(0, None, None), 0)),
            Index('multi-row', (slice(0, 2, 1), slice(0, None, None))),
        )
        self.assertEqual(self._actual.numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def _get_expected(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        jac = utils._compute_numerical_jacobian(
            self.func, xs, self._eps, self._dtype
        )
        return utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NM)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'func', 'xs'),
    (
        (
            '1d_in_1d_out',
            utils.square,
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]]),
        ),
        ('3d_in_3d_out', utils.square, np.random.rand(2, 3, 4)),
        ('multi_in_single_out', utils.square, np.random.rand(2, 3)),
    ),
)
class TestJacobianBatchFirst(unittest.TestCase):
    def setUp(self):
        self._dtype = (
            self.xs[0].dtype
            if isinstance(self.xs, typing.Sequence)
            else self.xs.dtype
        )
        self._eps = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("eps")
        )
        self._rtol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        self._atol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_jacobian(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=0)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index(
                'all',
                (
                    slice(0, None, None),
                    slice(0, None, None),
                    slice(0, None, None),
                ),
            ),
            Index('row', (slice(0, None, None), 0, slice(0, None, None))),
            Index('col', (slice(0, None, None), slice(0, None, None), 0)),
            Index(
                'batch',
                (slice(0, 2, None), slice(0, None, None), slice(0, None, None)),
            ),
            Index(
                'multi_row',
                (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None)),
            ),
        )
        self.assertEqual(self._actual[:].numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def test_jacobian_attribute_operator(self):
        # test for attribute operator "."
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        ys = (
            self.func(*xs) if isinstance(xs, typing.Sequence) else self.func(xs)
        )
        self._actual = paddle.autograd.jacobian(ys, xs, batch_axis=0)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index(
                'all',
                (
                    slice(0, None, None),
                    slice(0, None, None),
                    slice(0, None, None),
                ),
            ),
            Index('row', (slice(0, None, None), 0, slice(0, None, None))),
            Index('col', (slice(0, None, None), slice(0, None, None), 0)),
            Index(
                'batch',
                (slice(0, 2, None), slice(0, None, None), slice(0, None, None)),
            ),
            Index(
                'multi_row',
                (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None)),
            ),
        )
        self.assertEqual(self._actual.numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def _get_expected(self):
        xs = (
            [paddle.to_tensor(x, stop_gradient=False) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs, stop_gradient=False)
        )
        jac = utils._compute_numerical_batch_jacobian(
            self.func, xs, self._eps, self._dtype, False
        )
        jac = utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NBM)
        return utils._np_transpose_matrix_format(
            jac, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM
        )


class TestHessianNoBatch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (2, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = (
            config.TOLERANCE.get(self.dtype).get("second_order_grad").get("eps")
        )
        self.rtol = (
            config.TOLERANCE.get(self.dtype)
            .get("second_order_grad")
            .get("rtol")
        )
        self.atol = (
            config.TOLERANCE.get(self.dtype)
            .get("second_order_grad")
            .get("atol")
        )
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def func_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)

        self.x.stop_gradient = False
        y = func(self.x)
        hessian = paddle.autograd.hessian(y, self.x, batch_axis=None)
        np.testing.assert_allclose(
            hessian[:].numpy(), numerical_hessian, self.rtol, self.atol
        )

    def func_single_input_attribute_operator(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)

        self.x.stop_gradient = False
        y = func(self.x)
        hessian = paddle.autograd.hessian(y, self.x, batch_axis=None)
        np.testing.assert_allclose(
            hessian.numpy(), numerical_hessian, self.rtol, self.atol
        )

    def func_multi_input(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, y))

        numerical_hessian = utils._compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype
        )
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.hessian(
            func(self.x, self.y), [self.x, self.y], batch_axis=None
        )
        if isinstance(hessian, (tuple, list)):
            N_block = len(hessian)
            hessian = paddle.concat(
                [
                    paddle.concat(
                        [hessian[i][j][:] for j in range(N_block)], axis=1
                    )
                    for i in range(N_block)
                ],
                axis=0,
            )
        np.testing.assert_allclose(
            hessian.numpy(),
            numerical_hessian,
            rtol=self.rtol,
            atol=self.atol,
        )

    def func_allow_unused_true(self):
        def func(x, y):
            return paddle.sum(paddle.matmul(x, x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype
        )
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.autograd.hessian(
            func(self.x, self.y), [self.x, self.y], None
        )
        if isinstance(hessian, (tuple, list)):
            N_block = len(hessian)
            hessian = paddle.concat(
                [
                    paddle.concat(
                        [hessian[i][j][:] for j in range(N_block)], axis=1
                    )
                    for i in range(N_block)
                ],
                axis=0,
            )
        np.testing.assert_allclose(
            hessian.numpy(), numerical_hessian, self.rtol, self.atol
        )

    def func_create_graph_true(self):
        def func(x):
            return paddle.sum(F.sigmoid(x))

        numerical_hessian = utils._compute_numerical_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func(self.x), self.x, batch_axis=None)
        assert not hessian[:].stop_gradient
        np.testing.assert_allclose(
            hessian[:].numpy(), numerical_hessian, self.rtol, self.atol
        )

    def func_out_not_single(self):
        def func(x):
            return x * x

        with self.assertRaises(ValueError):
            x = paddle.ones([3])
            paddle.autograd.hessian(func(x), x, batch_axis=None)

    def test_all_cases(self):
        self.setUpClass()
        self.func_single_input()
        self.func_single_input_attribute_operator()
        self.func_multi_input()
        self.func_allow_unused_true()
        self.func_create_graph_true()
        self.func_out_not_single()


class TestHessianBatchFirst(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.x_shape = (5, 2)
        self.weight_shape = (2, 4)
        self.y_shape = (5, 2)
        self.nbatch, self.nrow = 5, 2
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = (
            config.TOLERANCE.get(self.dtype).get('second_order_grad').get('eps')
        )
        self.rtol = (
            config.TOLERANCE.get(self.dtype)
            .get('second_order_grad')
            .get('rtol')
        )
        self.atol = (
            config.TOLERANCE.get(self.dtype)
            .get('second_order_grad')
            .get('atol')
        )
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.weight.stop_gradient = False
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

    def func_single_input(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        expected = utils._compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )

        H = paddle.autograd.hessian(func(self.x), self.x, batch_axis=0)
        actual = utils._np_transpose_matrix_format(
            H[:].numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM
        )
        actual = actual.reshape((H.shape[1], -1))

        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_single_input_attribute_operator(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        expected = utils._compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )

        H = paddle.autograd.hessian(func(self.x), self.x, batch_axis=0)
        actual = utils._np_transpose_matrix_format(
            H.numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM
        )
        actual = actual.reshape((H.shape[1], -1))

        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_multi_input(self):
        def func(x, y):
            return paddle.matmul(x * x * y * y, self.weight)[:, 0:1]

        xs_len = 2
        expected = utils._compute_numerical_batch_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype
        )
        expected = np.reshape(
            np.array(expected),
            (xs_len, xs_len, self.nrow, self.nbatch, self.nrow),
        )
        expected = [list(row) for row in expected]
        expected = utils._np_concat_matrix_sequence(expected)

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        H = paddle.autograd.hessian(
            func(self.x, self.y), [self.x, self.y], batch_axis=0
        )
        if isinstance(H, (tuple, list)):
            H_t = []
            for i in range(len(H)):
                H_t.append(
                    paddle.concat(
                        [H[i][j][:] for j in range(len(H[0]))], axis=2
                    )
                )
            H_t = paddle.concat(H_t, axis=1)
            H = H_t

        actual = utils._np_transpose_matrix_format(
            H.numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM
        )

        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_allow_unused(self):
        def func(x, y):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        xs_len = 2
        expected = utils._compute_numerical_batch_hessian(
            func, [self.x, self.y], self.numerical_delta, self.np_dtype
        )
        expected = np.reshape(
            np.array(expected),
            (xs_len, xs_len, self.nrow, self.nbatch, self.nrow),
        )
        expected = [list(row) for row in expected]
        expected = utils._np_concat_matrix_sequence(expected)
        expected = utils._np_transpose_matrix_format(
            expected, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM
        )

        actual = paddle.autograd.hessian(
            func(self.x, self.y), [self.x, self.y], batch_axis=0
        )
        actual = paddle.concat(
            [
                paddle.concat([actual[i][j][:] for j in range(2)], axis=2)
                for i in range(2)
            ],
            axis=1,
        )

        np.testing.assert_allclose(
            actual, expected, rtol=self.rtol, atol=self.atol
        )

    def func_stop_gradient(self):
        def func(x):
            return paddle.matmul(x * x, self.weight)[:, 0:1]

        expected = utils._compute_numerical_batch_hessian(
            func, self.x, self.numerical_delta, self.np_dtype
        )

        x = self.x.clone()
        x.stop_gradient = True
        H = paddle.autograd.hessian(func(self.x), self.x, batch_axis=0)[:]
        actual = utils._np_transpose_matrix_format(
            H[:].numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM
        )
        actual = actual.reshape((H.shape[1], -1))

        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_out_not_single(self):
        def func(x):
            return x * x

        with self.assertRaises(ValueError):
            x = paddle.ones((3, 3))
            paddle.autograd.hessian(func(x), x, batch_axis=0)

    def test_all_cases(self):
        self.setUpClass()
        self.func_single_input()
        self.func_single_input_attribute_operator()
        self.func_multi_input()
        self.func_allow_unused()
        self.func_stop_gradient()
        self.func_out_not_single()


if __name__ == "__main__":
    np.random.seed(2022)
    unittest.main()
