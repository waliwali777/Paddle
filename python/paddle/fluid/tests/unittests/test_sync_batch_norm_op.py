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

from __future__ import print_function

import unittest
import numpy as np
import os
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler


class TestSyncBatchNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        #self.dtype = np.float64
        self.N = 32
        self.C = 16
        self.H = 64
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]

    def build_program(self,
                      place,
                      layout,
                      seed,
                      sync_bn=False,
                      only_forward=False):
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                    append_batch_size=False)
                conv = fluid.layers.conv2d(
                    input=data,
                    num_filters=8,  #32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
                    use_cudnn=False)
                if sync_bn:
                    bn = fluid.layers.sync_batch_norm(
                        conv,
                        param_attr=fluid.ParamAttr(name='bn_scale'),
                        bias_attr=fluid.ParamAttr(name='bn_bias'),
                        moving_mean_name='bn_moving_mean_sync',
                        moving_variance_name='bn_moving_variance_sync',
                        data_layout=layout,
                        is_test=only_forward)
                    sigmoid = fluid.layers.sigmoid(bn)
                    out = fluid.layers.reduce_sum(sigmoid)
                else:
                    bn = fluid.layers.batch_norm(
                        conv,
                        param_attr=fluid.ParamAttr(name='bn_scale'),
                        bias_attr=fluid.ParamAttr(name='bn_bias'),
                        moving_mean_name='bn_moving_mean_async',
                        moving_variance_name='bn_moving_variance_async',
                        data_layout=layout,
                        is_test=only_forward)
                    sigmoid = fluid.layers.sigmoid(bn)
                    out = fluid.layers.reduce_sum(sigmoid)
                    out = out / core.get_cuda_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, conv, bn]

    def compare(self, place, layout, only_forward):
        seed = 10
        #np.random.seed(5)
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        # Single-GPU, N = 32 per GPU
        main, startup, outs = self.build_program(place, layout, seed, False,
                                                 only_forward)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean_async', 'bn_moving_variance_async', 'bn_scale',
            'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_2@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        bn_fetches = exe.run(program=main,
                             feed={'input': data},
                             fetch_list=fetch_names)
        # Multi-GPUs, N = 32 per GPU
        main, startup, outs = self.build_program(place, layout, seed, True,
                                                 only_forward)
        exe = fluid.Executor(place)
        exe.run(startup)
        sync_fetch_names = [v.name for v in outs] + [
            'bn_moving_mean_sync', 'bn_moving_variance_sync', 'bn_scale',
            'bn_bias'
        ]
        if not only_forward:
            others = [
                'sync_batch_norm_0.tmp_0', 'sync_batch_norm_0.tmp_1',
                'bn_scale@GRAD', 'bn_bias@GRAD', 'sync_batch_norm_0.tmp_2@GRAD',
                'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        for nm in sync_fetch_names:
            fv = fluid.framework._get_var(str(nm), program=main)
            fv.persistable = True
        comp_prog = compiler.CompiledProgram(main).with_data_parallel(outs[
            0].name if not only_forward else None)
        sync_bn_fetches = exe.run(program=comp_prog,
                                  feed={'input': data},
                                  fetch_list=sync_fetch_names)

        for i in xrange(1, len(sync_bn_fetches)):
            bn_val = bn_fetches[i]
            sync_bn_val = sync_bn_fetches[i]
            if sync_bn_val.shape != bn_val.shape:
                sync_bn_val = sync_bn_val[:bn_val.shape[0]]
            self.assertTrue(
                np.allclose(
                    bn_val, sync_bn_val, atol=1e-3),
                "Output (" + fetch_names[i] + ") has diff. \n" + "\nBN     " +
                str(bn_val) + "\n" + "Sync BN " + str(sync_bn_val))

    def test_train(self):
        if not core.is_compiled_with_cuda():
            pass

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self.compare(place, layout, False)

    def test_infer(self):
        if not core.is_compiled_with_cuda():
            pass

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self.compare(place, layout, True)


if __name__ == '__main__':
    unittest.main()
