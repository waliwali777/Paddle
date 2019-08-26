# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import unittest
import paddle.fluid as fluid
from simple_nets import init_data, simple_fc_net
import os
os.environ['CPU_NUM'] = str(4)


class TestBackward(unittest.TestCase):
    def optimizer(self):
        learning_rate = fluid.layers.create_global_var(
            name="learning_rate",
            shape=[1],
            value=1.0,
            dtype='float32',
            persistable=True)
        optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
        return optimizer

    def check_backward(self, feed_dict):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = simple_fc_net()

            optimizer = self.optimizer()
            optimizer.minimize(loss)

            exe.run(program=startup)
            compiled_prog = fluid.compiler.CompiledProgram(
                main).with_data_parallel(loss_name=loss.name)

            exe.run(program=compiled_prog, feed=feed_dict)

    def test_backward(self):
        batch_size = 4
        img, label = init_data(batch_size, img_shape=[784], label_range=9)
        feed_dict = {
            'image': img,
            'label': label,
            'learning_rate': numpy.array([1.0]).astype("float32")
        }
        self.check_backward(feed_dict)


if __name__ == '__main__':
    unittest.main()
