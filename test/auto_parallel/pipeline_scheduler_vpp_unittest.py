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

import random
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()

PP_MESH_0 = auto.ProcessMesh([0])
PP_MESH_1 = auto.ProcessMesh([1])


class MyLinear(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        weight_attr=None,
    ):
        super().__init__()

        self.linear0 = nn.Linear(
            hidden_size, intermediate_size, weight_attr, bias_attr=None
        )
        self.linear1 = nn.Linear(
            intermediate_size, hidden_size, weight_attr, bias_attr=None
        )
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.linear0(input)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)

        return out


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()

        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )

        self.layers = nn.LayerList(
            [
                MyLinear(
                    hidden_size, intermediate_size, dropout_ratio, weight_attr
                )
                for _ in range(4)
            ]
        )

        self.linear = nn.Linear(hidden_size, 1, weight_attr, bias_attr=None)
        self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        self.layer_to_mesh = [PP_MESH_0, PP_MESH_1, PP_MESH_0, PP_MESH_1]

    def forward(self, input):
        out = self.norm(input)

        for i, layer in enumerate(self.layers):
            auto.shard_tensor(out, self.layer_to_mesh[i], [None, None])
            out = layer(out)

        out = self.linear(out)
        return out


def apply_pass(schedule_mode, acc_step):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = True
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = acc_step
    pipeline.vpp_degree = 2
    pipeline.vpp_seg_method = "MyLinear"

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class MyDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=1024).astype("float32")
        label = np.random.randint(0, 9, dtype="int64")
        return input, label

    def __len__(self):
        return self.num_samples


class TestVPPPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 4
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = MyDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2024 + paddle.distributed.get_rank())
        np.random.seed(2024 + paddle.distributed.get_rank())
        random.seed(2024 + paddle.distributed.get_rank())
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.base.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, schedule_mode, acc_step):
        reset_prog()

        strategy = apply_pass(schedule_mode, acc_step)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model = MLPLayer()
        loss = paddle.nn.CrossEntropyLoss()

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=self.rtol,
            atol=self.atol,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_pp_pass(self):
        # pp2-fthenb
        engine_fthenb = self.get_engine(schedule_mode="FThenB", acc_step=2)
        history_fthenb = engine_fthenb.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine_fthenb._strategy.pipeline.schedule_mode == "FThenB"

        # pp2-vpp
        engine_vpp_acc2 = self.get_engine(schedule_mode="VPP", acc_step=2)
        history_vpp_acc2 = engine_vpp_acc2.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine_vpp_acc2._strategy.pipeline.schedule_mode == "VPP"

        # pp2-1f1b
        engine_1f1b = self.get_engine(schedule_mode="1F1B", acc_step=4)
        history_1f1b = engine_1f1b.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine_1f1b._strategy.pipeline.schedule_mode == "1F1B"

        # pp2-vpp
        engine_vpp_acc4 = self.get_engine(schedule_mode="VPP", acc_step=4)
        history_vpp_acc4 = engine_vpp_acc4.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine_vpp_acc4._strategy.pipeline.schedule_mode == "VPP"

        if paddle.distributed.get_rank() == 1:
            losses_fthenb = np.array(history_fthenb.history["loss"])
            losses_vpp_acc2 = np.array(history_vpp_acc2.history["loss"])
            self.check_results(losses_fthenb, losses_vpp_acc2)

            losses_1f1b = np.array(history_1f1b.history["loss"])
            losses_vpp_acc4 = np.array(history_vpp_acc4.history["loss"])
            self.check_results(losses_1f1b, losses_vpp_acc4)


if __name__ == "__main__":
    unittest.main()
