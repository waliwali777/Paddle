# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import numpy as np
import pytest

import paddle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


paddle.enable_static()
np.random.seed(33)
main_program = paddle.static.Program()
startup_program = paddle.static.Program()
main_program.random_seed = 33
startup_program.random_seed = 33


@pytest.mark.softmax_with_cross_entropy
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_softmax_with_cross_entropyv2():
    with paddle.utils.unique_name.guard():
        # soft_label=True   use_softmax=True
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            x = paddle.static.data(name='x', shape=[48, 10], dtype='float32')
            y = paddle.static.data(name='y', shape=[48, 10], dtype='float32')
            y /= paddle.sum(y, axis=-1, keepdim=True)
            x.stop_gradient = False
            y.stop_gradient = True
            out = paddle.nn.functional.cross_entropy(
                x, y, soft_label=True, axis=-1, reduction="mean"
            )

            fetch_list = [out.name]
            g = paddle.static.gradients(out, [x, y])
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)

            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.randn(48, 10).astype('float32')
            y = np.random.uniform(0.1, 1.0, (48, 10)).astype('float32')
            output_dtu = exe.run(
                main_program, feed={"x": x, "y": y}, fetch_list=fetch_list
            )
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output_cpu = exec.run(
                main_program, feed={"x": x, "y": y}, fetch_list=fetch_list
            )
            print("output num:", len(output_dtu))
            for i in range(len(output_cpu)):
                print("------------")
                print(
                    np.allclose(
                        output_dtu[i], output_cpu[i], atol=1e-5, rtol=1e-5
                    )
                )
                print(fetch_list[i], output_dtu[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))
