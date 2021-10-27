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

from __future__ import print_function

import logging
import numpy as np
import os
import paddle
import shutil
import tempfile
import unittest

paddle.enable_static()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("paddle_with_cinn")


def set_cinn_flag(val):
    cinn_compiled = False
    try:
        paddle.set_flags({'FLAGS_use_cinn': val})
        cinn_compiled = True
    except ValueError:
        logger.warning("The used paddle is not compiled with CINN.")
    return cinn_compiled


def reader(limit):
    for _ in range(limit):
        yield np.random.random(size=(16, 1)).astype('float32')


def rand_data(loop_num=10):
    feed = []
    data = reader(loop_num)
    for i in range(loop_num):
        x = next(data)
        feed.append({'x': x})
    return feed


def build_program(main_program, startup_program):
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(name='x', shape=[None, 1], dtype='float32')

        prediction = paddle.static.nn.fc(x, 2)

        loss = paddle.mean(prediction)
        adam = paddle.optimizer.Adam()
        adam.minimize(loss)
    return prediction, loss


def do_test(dot_save_dir):
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    prediction, loss = build_program(main_program, startup_program)

    place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
    ) else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.debug_graphviz_path = os.path.join(dot_save_dir, "viz")
    compiled_program = paddle.static.CompiledProgram(
        main_program, build_strategy).with_data_parallel(loss_name=loss.name)

    feed = rand_data(10)
    for step in range(10):
        loss_v = exe.run(compiled_program,
                         feed=feed[step],
                         fetch_list=[loss.name],
                         return_merged=False)


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestParallelExecutorRunCinn(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dots_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_with_cinn(self):
        do_test(self.tmpdir)
        set_cinn_flag(False)


if __name__ == '__main__':
    do_test(tempfile.mkdtemp(prefix="dots_"))
    set_cinn_flag(False)
    # unittest.main()
