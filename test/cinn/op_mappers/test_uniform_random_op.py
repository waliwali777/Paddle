#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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


class TestUniformRandomOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [2, 3]
        self.min = -1.0
        self.max = 1.0
        self.seed = 10
        self.dtype = "float32"
        self.diag_num = 0
        self.diag_step = 1
        self.diag_val = 1.0

    def set_op_type(self):
        return "uniform_random"

    def set_op_inputs(self):
        return {}

    def set_op_attrs(self):
        return {
            "shape": self.shape,
            "min": self.min,
            "max": self.max,
            "seed": self.seed,
            "dtype": self.nptype2paddledtype(self.dtype),
            "diag_num": self.diag_num,
            "diag_step": self.diag_step,
            "diag_val": self.diag_val,
        }

    def set_op_outputs(self):
        return {'Out': [self.dtype]}

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # Uniform distribution.
        self.check_outputs_and_grads(max_relative_error=100)


class TestUniformRandomCase1(TestUniformRandomOp):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [2, 3, 4]
        self.min = -5.5
        self.max = 5.5
        self.seed = 10
        self.dtype = "float32"
        self.diag_num = 3
        self.diag_step = 3
        self.diag_val = 1.0


class TestUniformRandomCase2(TestUniformRandomOp):
    def init_input_data(self):
        self.feed_data = {}
        self.shape = [2, 3, 4]
        self.min = -10.0
        self.max = 10.0
        self.seed = 10
        self.dtype = "float64"
        self.diag_num = 24
        self.diag_step = 0
        self.diag_val = 1.0


if __name__ == "__main__":
    unittest.main()
