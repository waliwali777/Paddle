#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from op_mapper_test import OpMapperTest, logger
import paddle


class TestGatherOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([10, 12, 128, 128], 'float32'),
            'index': self.random([5], 'int32', 0, 10),
        }
        self.axis = 0

    def set_op_type(self):
        return "gather"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        index = paddle.static.data(
            name='index',
            shape=self.feed_data['index'].shape,
            dtype=self.feed_data['index'].dtype,
        )
        return {'X': [x], 'Index': [index]}

    def set_op_attrs(self):
        return {"axis": self.axis}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestGatherCase1(TestGatherOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([10, 12, 128, 128], 'float32'),
            'index': self.random([64], 'int32', 0, 128),
        }
        self.axis = 2


if __name__ == "__main__":
    unittest.main()
