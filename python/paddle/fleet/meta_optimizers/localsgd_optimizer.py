# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.optimizer import Momentum, SGD
from .meta_optimizer_base import MetaOptimizerBase
from . import CollectiveHelper


class LocalSGDOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(LocalSGDOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

        self.snapshot_key = '@SNAPSHOT'

    def _can_apply(self):
        if self.user_defined_strategy.localsgd:
            import paddle.fleet
            if fleet.trainer_num() <= 1:
                return False

            return isinstance(self.inner_opt, Momentum) \
                    or isinstance(self.inner_opt, SGD)
        return False

    def snapshot_name(self, param_name):
        return param_name + self.snapshot_key

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)

        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block

        self.nrings = 2
        self.wait_port = 1234
        collective_helper = CollectiveHelper(nrings, wait_port)
        # update startup program
        collective_helper.update_startup_program(startup_program)

        # update main program
        with program_guard(main_block.program):
            step = layers.autoincreased_step_counter(begin=0)
            last_step = layers.create_global_var(
                name="last_step",
                shape=[1],
                value=int(0),
                dtype='int64',
                persistable=True)

            global_lr = self.inner_opt._global_learning_rate()

            def initialize():
                loss_0 = layers.assign(loss)
                lr_0 = layers.assign(global_lr)

            layers.cond(step == 0, lambda: initialize)

            def communicate():
                ordered_param_snapshot = []
                ring_id = -1
                for idx, op in reversed(list(enumerate(block.ops))):
                    if collective_helper.is_update_op(op):
                        param = block.vars[op.input('Param')[0]]
                        if param.is_distributed:
                            continue

                        snapshot = block.create_var(
                            name=self.snapshot_name(param.name),
                            shape=param.shape,
                            persistable=True,
                            stop_gradient=True,
                            dtype=param.dtype)

                        block._insert_op(
                            idx + 1,
                            type='elementwise_sub',
                            inputs={'X': [snapshot],
                                    'Y': [param]},
                            outputs={'Out': [param]},
                            attrs={self.op_role_key: OpRole.Optimize})
                        block._insert_op(
                            idx + 2,
                            type='c_sync_calc_stream',
                            inputs={'X': param},
                            outputs={'Out': param},
                            attrs={self.op_role_key: OpRole.Optimize})
                        ring_id = (ring_id + 1) % self.nrings
                        block._insert_op(
                            idx + 3,
                            type='c_allreduce_sum',
                            inputs={'X': [param]},
                            outputs={'Out': [param]},
                            attrs={
                                'ring_id': ring_id,
                                self.op_role_key: OpRole.Optimize
                            })

                        ordered_param_snapshot.append((param, snapshot))

                for ring_id in range(self.nrings):
                    block.append_op(
                        type='c_sync_comm_stream',
                        inputs={'X': param},
                        outputs={'Out': param},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Optimize
                        })

                for param_snapshot in reversed(ordered_param_snapshot):
                    param = param_snapshot[0]
                    snapshot = param_snapshot[1]
                    block.append_op(
                        type='scale',
                        inputs={'X': [param]},
                        outputs={'Out': [param]},
                        attrs={
                            'scale': 1.0 / fleet.worker_num(),
                            self.op_role_key: OpRole.Optimize
                        })
                    block.append_op(
                        type='elementwise_sub',
                        inputs={'X': [snapshot],
                                'Y': [param]},
                        outputs={'Out': [param]},
                        attrs={self.op_role_key: OpRole.Optimize})
                    block.append_op(
                        type='assign',
                        inputs={'X': [param]},
                        outputs={'Out': [snapshot]},
                        attrs={self.op_role_key: OpRole.Optimize})

                delta_step = layers.cast(
                    layers.ceil(
                        layers.sqrt(lr_0 * loss / (global_lr * loss_0))),
                    dtype='int64')
                layers.assign(step, last_step)

            layers.cond(step - last_step == delta_step, communicate)

        return minimized
