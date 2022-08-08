#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import paddle
import unittest
import numpy as np
from op_test import OpTest
from paddle.nn.functional import kl_div


def kldiv_loss(x, target, reduction):
    output = target * (np.log(target) - x)
    loss = np.where(target >= 0, output, np.zeros_like(x))

    if reduction == "batchmean":
        if len(x.shape) > 0:
            return loss.sum() / x.shape[0]
        else:
            return loss.sum()
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()

    return loss


class TestKLDivLossOp(OpTest):

    def setUp(self):
        self.initTestCase()
        self.op_type = 'kldiv_loss'
        self.python_api = kl_div
        x = np.random.uniform(-10, 10, self.x_shape).astype('float64')
        target = np.random.uniform(-10, 10, self.x_shape).astype('float64')

        self.attrs = {"reduction": self.reduction}

        self.inputs = {
            'X': x,
            'Target': target,
        }
        loss = kldiv_loss(x, target, self.reduction)
        self.outputs = {'Loss': loss.astype('float64')}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'],
                        'Loss',
                        no_grad_set=set(["Target"]),
                        check_eager=True)

    def initTestCase(self):
        self.x_shape = (4, 5, 5)
        self.reduction = 'batchmean'


class TestKLDivLossOp2(TestKLDivLossOp):

    def initTestCase(self):
        self.x_shape = (3, 2, 7, 7)
        self.reduction = 'none'


class TestKLDivLossOp3(TestKLDivLossOp):

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 7, 9)
        self.reduction = 'mean'


class TestKLDivLossOp4(TestKLDivLossOp):

    def initTestCase(self):
        self.x_shape = (5, 20)
        self.reduction = 'sum'


class TestKLDivLossDygraph(unittest.TestCase):

    def run_kl_loss(self, reduction, shape=(5, 20)):
        x = np.random.uniform(-10, 10, shape).astype('float64')
        target = np.random.uniform(-10, 10, shape).astype('float64')
        gt_loss = kldiv_loss(x, target, reduction)

        with paddle.fluid.dygraph.guard():
            kldiv_criterion = paddle.nn.KLDivLoss(reduction)
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
            self.assertTrue(np.allclose(pred_loss.numpy(), gt_loss))

    def test_kl_loss_batchmean(self):
        self.run_kl_loss('batchmean')

    def test_kl_loss_batchmean_shape(self):
        self.run_kl_loss('batchmean', ())

    def test_kl_loss_mean(self):
        self.run_kl_loss('mean')

    def test_kl_loss_sum(self):
        self.run_kl_loss('sum')

    def test_kl_loss_none(self):
        self.run_kl_loss('none')

    def test_kl_loss_static_api(self):
        input = paddle.fluid.data(name='input', shape=[5, 20])
        label = paddle.fluid.data(name='label', shape=[5, 20])

        paddle.nn.functional.kl_div(input, label)
        paddle.nn.functional.kl_div(input, label, 'sum')
        paddle.nn.functional.kl_div(input, label, 'batchmean')


class TestKLDivLossTypePromotion(unittest.TestCase):

    def test_kl_div_promotion(self):

        with paddle.fluid.dygraph.guard():
            x1 = paddle.rand([5, 20], dtype='float32')
            target1 = paddle.rand([5, 20], dtype='float64')

            kldiv_criterion = paddle.nn.KLDivLoss()
            pred_loss1 = kldiv_criterion(x1, target1)

            x2 = paddle.rand([5, 20], dtype='float64')
            target2 = paddle.rand([5, 20], dtype='float32')
            pred_loss2 = paddle.nn.functional.kl_div(x2, target2)


class TestKlDivZeroDim(unittest.TestCase):

    def test_dygraph(self):
        paddle.disable_static()

        input = paddle.rand([10])
        label = paddle.rand([10])
        loss = paddle.nn.functional.kl_div(input, label)
        loss.backward()
        self.assertEqual(loss.shape, [])

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            input = paddle.static.data(name="input", shape=[10])
            label = paddle.static.data(name="label", shape=[10])
            out = paddle.nn.functional.kl_div(input, label)
            paddle.static.append_backward(out)

            input_np = np.random.random([10]).astype('float32')
            label_np = np.random.random([10]).astype('float32')
            exe = paddle.static.Executor()
            result = exe.run(paddle.static.default_main_program(),
                             feed={
                                 "input": input_np,
                                 "label": label_np
                             },
                             fetch_list=[out])

            self.assertEqual(result[0].shape, ())

if __name__ == "__main__":
    unittest.main()
