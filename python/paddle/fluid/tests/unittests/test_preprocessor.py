# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.v2 as paddle
import paddle.v2.dataset.mnist as mnist


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(mnist.train(), batch_size=32)
            feeder = fluid.DataFeeder(
                feed_list=[  # order is image and label
                    fluid.layers.data(
                        name='image', shape=[784]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            self.num_batches = fluid.recordio_writer.convert_reader_to_recordio_file(
                './mnist_for_preprocessor_test.recordio', reader, feeder)

    def test_main(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_file = fluid.layers.io.open_recordio_file(
                './mnist_for_preprocessor_test.recordio',
                shapes=[[-1, 784], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            preprocessor = fluid.layers.io.Preprocessor(reader=data_file)
            with preprocessor.block():
                img, lbl = preprocessor.inputs()
                img_out = img / 2
                lbl_out = lbl + 1
                preprocessor.outputs(img_out, lbl_out)

            img_before, lbl_before = fluid.layers.io.read_file(data_file)
            img_after, lbl_after = fluid.layers.io.read_file(preprocessor())

            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            else:
                place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            for _ in range(5):
                img_b, lbl_b, img_a, lbl_a = exe.run(
                    fetch_list=[img_before, lbl_before, img_after, lbl_after])

            self.assertEqual(img_b / 2, img_a)
            self.assertEqual(lbl_b + 1, lbl_a)
