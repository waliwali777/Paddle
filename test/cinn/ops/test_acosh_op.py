# Copyright (c) 2023 CINN Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle
from paddle.cinn.common import is_compiled_with_cuda
from paddle.cinn.frontend import NetBuilder


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestAcoshOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            low=2,
            high=100,
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = paddle.acosh(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("acosh")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )

        out = builder.acosh(x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestAcoshCase1(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestAcoshCase1"
        self.cls = TestAcoshOp
        self.inputs = [{"x_shape": [512, 256]}]
        self.dtypes = [
            {"x_dtype": "float32"},
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = []


class TestAcoshCase2(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestAcoshCase2"
        self.cls = TestAcoshOp
        self.inputs = [
            {"x_shape": [1]},
            {"x_shape": [1024]},
            {"x_shape": [512, 256]},
            {"x_shape": [128, 64, 32]},
            {"x_shape": [128, 2048, 32]},
            {"x_shape": [16, 8, 4, 2]},
            {"x_shape": [1, 1, 1, 1]},
            {"x_shape": [16, 8, 4, 2, 1]},
        ]
        self.dtypes = [{"x_dtype": "float32"}]
        self.attrs = []


if __name__ == "__main__":
    TestAcoshCase1().run()
    TestAcoshCase2().run()
