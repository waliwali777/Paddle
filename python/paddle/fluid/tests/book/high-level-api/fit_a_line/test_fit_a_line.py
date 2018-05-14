#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import contextlib
import numpy
import unittest

# train reader
BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)


# train
def linear(use_cuda, save_dirname, is_local):
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')

    global y_predict
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    loss = fluid.layers.cross_entropy(y_predict, y)
    return loss


def train(use_cuda, save_dirname, is_local):
    trainer = fluid.Trainer(linear, optimizer=fluid.optimizer.SGD())

    def event_handler(event):
        if isinstance(event, fluid.EndIteration):
            print event.metrics

        elif isinstance(event, fluid.EndPass):
            test_metrics = trainer.test(
                reader=paddle.dataset.uci_housing.test())
            print test_metrics

            if test_metrics[0] < 10.0:
                if save_dirname is not None:
                    fluid.io.save_inference_model(save_dirname, ['x'],
                                                  [y_predict])
            return

    trainer.train(reader=paddle.dataset.uci_housing.train(), num_pass=100)


# infer
def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(param_path=save_dirname, place=place)

    batch_size = 10
    tensor_x = numpy.random.uniform(0, 10, [batch_size, 13]).astype("float32")

    results = inferencer.infer({'x': tensor_x})
    print("infer results: ", results[0])


def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "fit_a_line.inference.model"

    train(use_cuda, save_dirname, is_local)
    infer(use_cuda, save_dirname)


class TestFitALine(unittest.TestCase):
    def test_cpu(self):
        with self.program_scope_guard():
            main(use_cuda=False)

    def test_cuda(self):
        with self.program_scope_guard():
            main(use_cuda=True)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
