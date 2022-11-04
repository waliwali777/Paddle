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

import unittest

import paddle
import numpy as np
import paddle.distributed as dist
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear

paddle.seed(1024)
np.random.seed(2021)

batch = 1
in_dim = 10
out_dim = 20


class SimpleNet(fluid.Layer):
    def __init__(self, train_id):
        super().__init__()
        self.w1 = self.create_parameter(
            shape=[in_dim, out_dim], dtype="float32"
        )
        self.w2 = self.create_parameter(
            shape=[in_dim, out_dim], dtype="float32"
        )
        self.share_net = Linear(out_dim, 1)

        self.unused_param = self.create_parameter(
            shape=[out_dim, in_dim], dtype="float32"
        )

        # just for test sync_params_buffers
        self.register_buffer("queue", paddle.randn([10, 5]))
        self.queue = paddle.nn.functional.normalize(self.queue, axis=0)
        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

        self.trainer_id = train_id

    def forward(self, x):
        is_use = (
            paddle.equal_all(x, paddle.ones(shape=(batch, in_dim))).numpy()[0]
            and self.trainer_id == 1
        )

        if is_use:
            tmp = paddle.matmul(x, self.w1)
        else:
            tmp = paddle.matmul(x, self.w2)

        return self.share_net(tmp)


class TestDistTraning(unittest.TestCase):
    def test_multiple_gpus(self):
        self.trainer_id = dist.get_rank()
        dist.init_parallel_env()

        model_a = SimpleNet(self.trainer_id)
        model_b = SimpleNet(self.trainer_id)

        state_dict = model_a.state_dict()
        model_b.set_state_dict(state_dict)

        model_a = paddle.DataParallel(model_a, find_unused_parameters=True)
        model_b = paddle.DataParallel(model_b, find_unused_parameters=True)

        ones_input = paddle.ones(shape=(batch, in_dim))
        ones_input.stop_gradient = True

        for step_id in range(1, 31):
            random_input = paddle.rand(shape=(batch, in_dim))
            random_input.stop_gradient = True

            if step_id % 5 != 0:
                with model_a.no_sync():
                    self.dp_layer(
                        step_id, model_a, model_b, random_input, ones_input
                    )
            else:
                self.dp_layer(
                    step_id, model_a, model_b, random_input, ones_input
                )

                self.check_gradient(model_a.parameters())
                self.check_gradient(model_b.parameters())

                self.check_acc(model_a._layers.w1.grad, model_b._layers.w1.grad)
                self.check_acc(model_a._layers.w2.grad, model_b._layers.w2.grad)

                model_a.clear_gradients()
                model_b.clear_gradients()

    def dp_layer(self, step_id, model_a, model_b, random_input, ones_input):
        if step_id % 2 == 0:
            out_a = model_a(random_input)
            out_b = model_b(random_input)
        else:
            out_a = model_a(ones_input)
            out_b = model_b(ones_input)
        out_a.sum().backward()
        out_b.sum().backward()

    def check_acc(self, grad, acc_grad):
        grad = grad.numpy() if grad is not None else None
        acc_grad = acc_grad.numpy() if acc_grad is not None else None
        return np.testing.assert_allclose(grad, acc_grad, rtol=1e-6)

    def print_trainer_0(self, *args):
        if self.trainer_id == 0:
            print(*args)

    def broadcast_param(self, param, root):
        paddle.distributed.broadcast(param, root)
        return param

    def check_gradient(self, params):
        other_param = []
        for param in params:
            if param.trainable and (param._grad_ivar() is not None):
                grad = param._grad_ivar()
                other_grad = self.broadcast_param(grad.clone(), root=1)
                if self.trainer_id == 0:
                    np.testing.assert_allclose(other_grad.numpy(), grad.numpy())


if __name__ == '__main__':
    unittest.main()
