#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import yaml
import unittest
from paddle.distributed.auto_parallel.strategy import Strategy


class TestStrategy(unittest.TestCase):

    def test_default_config(self):
        strategy = Strategy()

        recompute = strategy.recompute
        self.assertEqual(recompute.enabled, False)
        self.assertEqual(recompute.checkpoints, None)

        amp = strategy.amp
        self.assertEqual(amp.enabled, False)
        self.assertAlmostEqual(amp.init_loss_scaling, 32768.0)
        self.assertEqual(amp.incr_every_n_steps, 1000)
        self.assertEqual(amp.decr_every_n_nan_or_inf, 2)
        self.assertAlmostEqual(amp.incr_ratio, 2.0)
        self.assertAlmostEqual(amp.decr_ratio, 0.8)
        self.assertEqual(amp.use_dynamic_loss_scaling, True)
        self.assertEqual(amp.custom_black_list, None)
        self.assertEqual(amp.custom_white_list, None)
        self.assertEqual(amp.custom_black_varnames, None)
        self.assertEqual(amp.use_pure_fp16, False)
        self.assertEqual(amp.use_fp16_guard, True)
        self.assertEqual(amp.use_optimizer_fp16, False)

        sharding = strategy.sharding
        self.assertEqual(sharding.enabled, False)
        self.assertEqual(sharding.stage, 1)
        self.assertEqual(sharding.sharding_degree, 8)
        self.assertAlmostEqual(sharding.segment_broadcast_MB, 32.0)
        self.assertEqual(sharding.enable_tuning, False)

        gradient_merge = strategy.gradient_merge
        self.assertEqual(gradient_merge.enabled, False)
        self.assertEqual(gradient_merge.k_steps, 1)
        self.assertEqual(gradient_merge.avg, True)

    def test_modify_config(self):
        strategy = Strategy()

        recompute = strategy.recompute
        recompute.enabled = True
        recompute.checkpoinits = ["x"]
        self.assertEqual(recompute.enabled, True)
        self.assertEqual(recompute.checkpoinits, ["x"])

        amp = strategy.amp
        amp.enabled = True
        amp.init_loss_scaling = 16384.0
        amp.incr_every_n_steps = 2000
        amp.decr_every_n_nan_or_inf = 4
        amp.incr_ratio = 4.0
        amp.decr_ratio = 0.4
        amp.use_dynamic_loss_scaling = False
        amp.custom_white_list = ["x"]
        amp.custom_black_list = ["y"]
        amp.custom_black_varnames = ["z"]
        amp.use_pure_fp16 = True
        amp.use_fp16_guard = False
        amp.use_optimizer_fp16 = True
        self.assertEqual(amp.enabled, True)
        self.assertAlmostEqual(amp.init_loss_scaling, 16384.0)
        self.assertEqual(amp.incr_every_n_steps, 2000)
        self.assertEqual(amp.decr_every_n_nan_or_inf, 4)
        self.assertAlmostEqual(amp.incr_ratio, 4.0)
        self.assertAlmostEqual(amp.decr_ratio, 0.4)
        self.assertEqual(amp.use_dynamic_loss_scaling, False)
        self.assertEqual(amp.custom_white_list, ["x"])
        self.assertEqual(amp.custom_black_list, ["y"])
        self.assertEqual(amp.custom_black_varnames, ["z"])
        self.assertEqual(amp.use_pure_fp16, True)
        self.assertEqual(amp.use_fp16_guard, False)
        self.assertEqual(amp.use_optimizer_fp16, True)

        sharding = strategy.sharding
        sharding.enabled = True
        sharding.stage = 2
        sharding.sharding_degree = 2
        sharding.segment_broadcast_MB = 64.0
        sharding.enable_tuning = True
        self.assertEqual(sharding.enabled, True)
        self.assertEqual(sharding.stage, 2)
        self.assertEqual(sharding.sharding_degree, 2)
        self.assertAlmostEqual(sharding.segment_broadcast_MB, 64.0)
        self.assertEqual(sharding.enable_tuning, True)

        gradient_merge = strategy.gradient_merge
        gradient_merge.enabled = True
        gradient_merge.k_steps = 4
        gradient_merge.avg = False
        self.assertEqual(gradient_merge.enabled, True)
        self.assertEqual(gradient_merge.k_steps, 4)
        self.assertEqual(gradient_merge.avg, False)

    def test_file_config(self):
        yaml_data = """
        amp:
            custom_black_list:
            - y
            custom_black_varnames:
            - z
            custom_white_list:
            - x
            decr_every_n_nan_or_inf: 4
            decr_ratio: 0.4
            enabled: true
            incr_every_n_steps: 2000
            incr_ratio: 4.0
            init_loss_scaling: 16384.0
            use_dynamic_loss_scaling: false
            use_fp16_guard: false
            use_optimizer_fp16: true
            use_pure_fp16: true
        auto_mode: semi
        gradient_merge:
            avg: false
            enabled: true
            k_steps: 4
        recompute:
            checkpoints: null
            enabled: true
        sharding:
            enable_tuning: true
            enabled: true
            segment_broadcast_MB: 64.0
            sharding_degree: 2
            stage: 2
        """
        yaml_path = "./strategy.yml"
        yaml_dict = yaml.load(yaml_data, Loader=yaml.Loader)
        with open(yaml_path, 'w') as outfile:
            yaml.dump(yaml_dict, outfile, default_flow_style=False)

        strategy = Strategy(yaml_path)

        self.assertEqual(yaml_dict, strategy.to_dict())

        # Remove the created file
        if os.path.exists(yaml_path):
            os.remove(yaml_path)


if __name__ == '__main__':
    unittest.main()
