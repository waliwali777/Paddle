#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        x = np.random.uniform(size=[3, 4, 2, 2])
        target = np.random.uniform(size=[3, 4, 2, 2])
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "target": target.astype(np.float32),
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "target": target.astype(np.float16),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {
            'delta': 1.0,
        }

    @IPUOpTest.static_graph
    def build_model(self, on_ipu):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype="float32"
        )
        target = paddle.static.data(
            name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32'
        )
        out = paddle.fluid.layers.huber_loss(x, target, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model(self.is_ipu_mode(m))
                self.run_model(m)
        self.check()


class TestCase1(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            'delta': 0.5,
        }


class TestCase2(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            'delta': 0.0,
        }


if __name__ == "__main__":
    unittest.main()
