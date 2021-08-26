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


class TrtConvertReshapeTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "reshape",
            "op_inputs": {
                "X": ["input_data"]
            },
            "op_outputs": {
                "Out": ["shape_output_data"]
            },
            "op_attrs": {
                "shape": [[1, 6, 8], [1, 2, 4, 6]]
            }
        }]
        self.batch_size_set = [1, 1, 1]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        input_data = TensorConfig(shape=[1, 48])
        self.program_weights = {"conv2d_weight": filter}
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["shape_output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)


class TrtConvertDynamicReshapeTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "reshape",
            "op_inputs": {
                "X": ["input_data"]
            },
            "op_outputs": {
                "Out": ["shape_output_data"]
            },
            "op_attrs": {
                "shape": [[2, 24], [6, 8]]
            }
        }]
        self.batch_size_set = [1, 1, 1]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        input_data = TensorConfig(shape=[1, 12, 4])
        self.program_weights = {"conv2d_weight": filter}
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["shape_output_data"]

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"input_data": [1, 12, 4]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 12, 4]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 12, 4]}
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"input_data": [1, 12, 4]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 12, 4]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 12, 4]}
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)


class TrtConvertReshapeInputShapeTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "reshape",
            "op_inputs": {
                "X": ["input_data"],
                "Shape": ["shape_data"],
            },
            "op_outputs": {
                "Out": ["shape_output_data"]
            },
            "op_attrs": {
                "shape": [[1, 48]]
            }
        }]
        self.batch_size_set = [1, 1, 1]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        input_data = TensorConfig(shape=[1, 2, 4, 6])
        shape_data = TensorConfig(
            shape=[1, 2],
            dtype='int32',
            data=np.array([1, 48]).astype(np.int32))
        self.program_weights = {
            "conv2d_weight": filter,
            "shape_data": shape_data
        }
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["shape_output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"input_data": [1, 2, 4, 6]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 2, 4, 6]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 2, 4, 6]}
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"input_data": [1, 2, 4, 6]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 2, 4, 6]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 2, 4, 6]}
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-2)


class TrtConvertReshapeInputShapeTensorTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "reshape",
            "op_inputs": {
                "X": ["input_data"],
                "ShapeTensor": ["shapeT1_data", "shapeT2_data"]
            },
            "op_outputs": {
                "Out": ["shape_output_data"]
            },
            "op_attrs": {
                "shape": [[1, 48]]
            }
        }]
        self.batch_size_set = [1, 1, 1]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        input_data = TensorConfig(shape=[1, 2, 4, 6])
        shapeT1_data = TensorConfig(
            shape=[1], dtype='int32', data=np.array([2]).astype(np.int32))
        shapeT2_data = TensorConfig(
            shape=[1], dtype='int32', data=np.array([24]).astype(np.int32))
        self.program_weights = {
            "conv2d_weight": filter,
            "shapeT1_data": shapeT1_data,
            "shapeT2_data": shapeT2_data
        }
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["shape_output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"input_data": [1, 2, 4, 6]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 2, 4, 6]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 2, 4, 6]}
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"input_data": [1, 2, 4, 6]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 2, 4, 6]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 2, 4, 6]}
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-2)


if __name__ == "__main__":
    unittest.main()
