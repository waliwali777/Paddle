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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest
from program_config import TensorConfig
import numpy as np
import paddle.inference as paddle_infer
import unittest


class TrtConvertSoftmaxTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "softmax",
            "op_inputs": {
                "X": ["softmax_channel_input_data"]
            },
            "op_outputs": {
                "Out": ["softmax_channel_output_data"]
            },
            "op_attrs": {
                "axis": [1, 2, 3]
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        softmax_channel_input_data = TensorConfig(
            shape=[1, 3, 64, 64], dtype="float32")
        self.program_weights = {}
        self.program_inputs = {
            "softmax_channel_input_data": softmax_channel_input_data
        }
        self.program_outputs = ["softmax_channel_output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {
            "softmax_channel_input_data": [1, 3, 32, 32]
        }
        self.dynamic_shape.max_input_shape = {
            "softmax_channel_input_data": [4, 3, 64, 64]
        }
        self.dynamic_shape.opt_input_shape = {
            "softmax_channel_input_data": [1, 3, 64, 64]
        }
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {
            "softmax_channel_input_data": [1, 3, 32, 32]
        }
        self.dynamic_shape.max_input_shape = {
            "softmax_channel_input_data": [4, 3, 64, 64]
        }
        self.dynamic_shape.opt_input_shape = {
            "softmax_channel_input_data": [1, 3, 64, 64]
        }
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)


class DynamicTrtConvertSoftmaxTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "softmax",
            "op_inputs": {
                "X": ["softmax_channel_input_data"]
            },
            "op_outputs": {
                "Out": ["softmax_channel_output_data"]
            },
            "op_attrs": {
                "axis": [0]
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        softmax_channel_input_data = TensorConfig(
            shape=[1, 3, 64, 64], dtype="float32")
        self.program_weights = {}
        self.program_inputs = {
            "softmax_channel_input_data": softmax_channel_input_data
        }
        self.program_outputs = ["softmax_channel_output_data"]

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {
            "softmax_channel_input_data": [1, 3, 32, 32]
        }
        self.dynamic_shape.max_input_shape = {
            "softmax_channel_input_data": [4, 3, 64, 64]
        }
        self.dynamic_shape.opt_input_shape = {
            "softmax_channel_input_data": [1, 3, 64, 64]
        }
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {
            "softmax_channel_input_data": [1, 3, 32, 32]
        }
        self.dynamic_shape.max_input_shape = {
            "softmax_channel_input_data": [4, 3, 64, 64]
        }
        self.dynamic_shape.opt_input_shape = {
            "softmax_channel_input_data": [1, 3, 64, 64]
        }
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)


if __name__ == "__main__":
    unittest.main()
