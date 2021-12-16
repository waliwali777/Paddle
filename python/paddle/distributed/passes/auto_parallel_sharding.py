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

from functools import reduce
from collections import OrderedDict
import numpy as np

import paddle
from paddle.framework import core
from paddle.fluid import unique_name
from .pass_base import PassBase, register_pass
from paddle.distributed.fleet.meta_optimizers.common import is_backward_op, is_optimizer_op
from paddle.distributed.auto_parallel.process_group import get_world_process_groups, new_process_group
from paddle.distributed.auto_parallel.operators.common import is_parameter_related
from paddle.distributed.auto_parallel.utils import _get_comm_group, naive_set_dist_op_attr_for_program_by_mesh_and_mapping

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
_skip_ops = ['create_py_reader', 'create_double_buffer_reader', 'read', 'slice']
# update here to support new optimizers
_supported_optimizer_type = [
    "adam", "adamax", "adamw", "decayed_adagrad", "momentum", "dgc_momentum",
    "lars_momentum", "merged_momentum", "lamb", "sgd"
]
# FIXME
global_processes = get_world_process_groups().ranks


def inference_data_parallel_group_for_operator(rank_id, op, dist_context):

    dp_group = None
    for input_name in op.input_arg_names:
        if not is_parameter_related(input_name, op.block):
            dist_attr = dist_context.get_op_dist_attr_for_program(op)
            process_mesh = dist_attr.process_mesh
            input_dim_mapping = dist_attr.get_input_dims_mapping(input_name)

            mesh_shape = process_mesh.topology
            # TODO replace with specific batch size dimension
            batch_size_axis = input_dim_mapping[0]
            if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                group_ranks = _get_comm_group(process_mesh.processes,
                                              process_mesh.topology,
                                              batch_size_axis, rank_id)
                dp_group = new_process_group(group_ranks)
                break

    return dp_group


# TODO (JZ-LIANG) merge forward & backward pass 
# NOTE we add the "auto_parallel" prefix to the pass in order to 
# indicate that this pass should obey some constrains by auto_parallel
# for example all ops and vars should has dist attr before and after pass
# should use dist op instead of custom comm op 
@register_pass("auto_parallel_sharding")
class ShardingPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()
        self.set_attr("dist_context", None)
        self.set_attr("stage", None)
        self.set_attr("sharding_degree", None)
        self.set_attr("params_grads", [])
        self.set_attr("global_rank", -1)
        self.dp_groups = set()
        self.sharding_infos = []
        self.var2sharding_info = {}

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False

        if self.get_attr("stage") not in [1, 2, 3]:
            return False
        if (not isinstance(self.get_attr("sharding_degree"),
                           int)) or self.get_attr("sharding_degree") <= 1:
            return False
        if len(self.get_attr("params_grads")) <= 0:
            return False
        if (not isinstance(self.get_attr("global_rank"),
                           int)) or self.get_attr("global_rank") < 0:
            return False

        return True

    def _check_conflict(self, other_pass):
        return True

    # NOTE: why ShardingPass can override apply_single_impl instead of 
    # apply_impl? AMP is an optimization pass for serial program, 
    # in distributed scenario, all ranks should have the same modification.
    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self.sharding_world_size = int(self.get_attr("sharding_degree"))
        self.stage = int(self.get_attr("stage"))
        self.global_rank = int(self.get_attr("global_rank"))
        params_grads = self.get_attr("params_grads")

        self._build_sharding_group(main_program, params_grads)
        self._shard_optimizer(main_program, startup_program, params_grads,
                              context)

    def _build_sharding_group(self, main_program, params_grads):

        self._collective_data_parallel_groups(main_program)
        self._build_sharding_info(params_grads)

    def _collective_data_parallel_groups(self, main_program):

        main_block = main_program.global_block()

        for op in main_block.ops:
            if op.type in _skip_ops:
                continue
            group = inference_data_parallel_group_for_operator(
                self.global_rank, op, self._dist_context)
            if group is not None:
                self.dp_groups.add(group)

        # TODO allow more than one dp groups in network, support more general distribution 
        # genetated by auto search
        if len(self.dp_groups) != 1:
            raise NotImplementedError(
                "So far Only and Exactly one data parallel group in network are supported, but got [{}] different data parallel groups".
                format(len(groups)))

    def _build_sharding_info(self, params_grads):

        for dp_group in self.dp_groups:

            assert dp_group.nranks >= self.sharding_world_size, "sharding world size [{}] should not larger than dp world size [{}]".format(
                self.sharding_world_size, dp_group.nranks)
            assert dp_group.nranks % self.sharding_world_size == 0, "sharding world size [{}] should be divisible by dp world size [{}]".format(
                self.sharding_world_size, dp_group.nranks)
            assert self.global_rank in dp_group.ranks, "current ranks [{}] does NOT belong to the data parallel group [{}]".format(
                self.global_rank, dp_group.ranks)
            assert len(
                params_grads
            ) >= self.sharding_world_size, "number of parameters [{}] is not enough to be shard among [{}] ranks".format(
                len(params_grads), self.sharding_world_size)

            # sharding hybrid data parallel: partial sharding param within 
            if dp_group.nranks > self.sharding_world_size:
                partial_group = get_partial_group(
                    dp_group.ranks, self.sharding_world_size, self.global_rank)
                group = new_process_group(partial_group)
            else:
                group = dp_group

            # TODO when support multiple dp groups in future, should group param and bind them to corresponding dp group
            params_in_group = [p for p, g in params_grads]
            assert len(params_in_group) == len(set(
                params_in_group)), "found duplicated param in params_grads"
            sharding_info = ShardingInfo(group, self.global_rank,
                                         params_in_group)
            self.sharding_infos.append(sharding_info)
            for param in params_in_group:
                self.var2sharding_info[param.name] = sharding_info

    def _shard_optimizer(self, main_program, startup_program, params_grads,
                         pass_context):
        """
        sharding all optimizer related ops and vars, include:
        gradient clip ops & vars
        weight decay ops & vars
        optimizer ops and states
        """
        main_block = main_program.global_block()
        startup_block = startup_program.global_block()

        # remove dependencies of gradient that not belong to local rank
        # allow the gradinet to be GC in time

        # self._shard_amp_backward(main_block, pass_context)
        # self._shard_weight_decay(main_block)
        # self._shard_gradient_clip(main_block)
        # self._shard_weight_decay(main_block)

        self._shard_optimizer_ops_and_states(main_block, startup_block)

    def _shard_amp_backward(self, main_block, pass_context):
        """
        shard the amp relatd backward logic among group:
        gradient cast 
        check_nan_inf
        unscale_gradient
        """
        # if not _is_amp_applied(pass_context):
        #     return 
        for idx, op in reversed(list(enumerate(main_block.ops))):

            # shard amp related param_grad cast
            if is_amp_related_param_grad_cast_op(main_block, op):
                output_name = op.output_arg_names[0]
                param_name = output_name[:output_name.find("@")]
                if not self._is_parameter_in_local_shard(param_name):
                    main_block._remove_op(idx, sync=False)
                    main_block._remove_var(output_name, sync=False)

            # shard check nan inf
            elif op.type in ["check_finite_and_unscale", "update_loss_scaling"]:
                reversed_x = []
                for input_name in op.desc.input('X'):
                    param_name = input_name[:input_name.find("@")]

                    if self._is_parameter_in_local_shard(param_name):
                        reversed_x.append(input_name)
                op.desc.set_input('X', reversed_x)
                op.desc.set_output('Out', reversed_x)

        main_block._sync_with_cpp()

    def _shard_gradient_clip(self, main_block):
        # TODO support calculate global norm with tensor parallelism

        is_clip_grad_by_global_norm = False
        for idx, op in list(enumerate(main_block.ops)):
            if not _is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                is_clip_grad_by_global_norm = True
                break
        if not is_clip_grad_by_global_norm:
            return

        removed_op_idx = set()
        removed_tmp_var = set()
        for idx, op in list(enumerate(main_block.ops)):
            if not _is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                reserved_vars = []
                for input_name in op.input_arg_names:
                    if input_name not in removed_tmp_var:
                        reserved_vars.append(input_name)
                op.desc.set_input("X", reserved_vars)

                sum_op_output = op.desc.output_arg_names()[0]
                for i, sharding_info in enumerate(self.sharding_infos):
                    new_op = main_block._insert_op(
                        idx + i,
                        type='c_allreduce_sum',
                        inputs={'X': [sum_op_output]},
                        outputs={'Out': [sum_op_output]},
                        attrs={
                            'ring_id': sharding_info.group.id,
                            'op_namescope': "/gradient_clip_model_parallelism",
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize,
                        })
                    dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                        main_block.var(sum_op_output))
                    assert dist_attr is not None
                    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                        new_op, dist_attr.process_mesh, dist_attr.dims_mapping,
                        self._dist_context)
                break
            for input_name in op.input_arg_names:
                param_name = input_name[:input_name.find("@GRAD")]
                if not self._is_parameter_in_local_shard(param_name):
                    removed_op_idx.add(idx)
                    for output_name in op.output_arg_names:
                        removed_tmp_var.add(output_name)

        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not _is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                main_block._remove_op(idx, sync=False)

        for var_name in removed_tmp_var:
            main_block._remove_var(var_name, sync=False)

        main_block._sync_with_cpp()

    def _shard_weight_decay(self, main_block):

        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not _is_weight_decay_op(op):
                continue
            else:
                raise NotImplementedError(
                    "weight decay is NOT supported by now")
        main_block._sync_with_cpp()

    def _shard_optimizer_ops_and_states(self, main_block, startup_block):

        should_removed_optimizer_states = []
        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not is_optimizer_op(op):
                break

            if op.type in _supported_optimizer_type:
                assert "Param" in op.input_names
                assert len(op.input("Param")) == 1
                param_name = op.input("Param")[0]
                if not self._is_parameter_in_local_shard(param_name):
                    should_removed_optimizer_states.extend([
                        varname for varname in op.output_arg_names
                        if varname != param_name
                    ])
                    main_block._remove_op(idx, sync=False)

        for idx, op in reversed(list(enumerate(startup_block.ops))):
            if len(op.output_arg_names) == 1 and op.output_arg_names[
                    0] in should_removed_optimizer_states:
                startup_block._remove_op(idx, sync=False)

        for varname in should_removed_optimizer_states:
            if main_block.has_var(varname):
                main_block._remove_var(varname, sync=False)
            if startup_block.has_var(varname):
                startup_block._remove_var(varname, sync=False)

        if self.stage <= 2:
            self._insert_not_overlap_broadcasts(main_block, startup_block)

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def _insert_not_overlap_broadcasts(self, main_block, startup_block):

        for sharding_info in self.sharding_infos:
            for param in sharding_info.params:
                assert main_block.has_var(param.name)
                assert startup_block.has_var(param.name)
                for block, role in zip([main_block, startup_block],
                                       [OpRole.Optimize, OpRole.Forward]):
                    new_op = block.append_op(
                        type='c_broadcast',
                        inputs={'X': param},
                        outputs={'Out': param},
                        attrs={
                            'ring_id': sharding_info.group.id,
                            'root': sharding_info.param2rank[param],
                            'use_calc_stream': True,
                            OP_ROLE_KEY: role
                        })
                    param_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                        param)
                    assert param_dist_attr is not None
                    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                        new_op, param_dist_attr.process_mesh,
                        param_dist_attr.dims_mapping, self._dist_context)

    def _is_parameter_in_local_shard(self, param_name):
        assert param_name in self.var2sharding_info
        sharding_info = self.var2sharding_info[param_name]
        return sharding_info.is_in_local_shard(param_name)


def get_partial_group(origin_group, partial_group_size, rank):
    chunk_groups = chunks(origin_group, partial_group_size)
    partial_group = None
    for ranks in chunk_groups:
        if rank in ranks:
            partial_group = ranks[:]
    assert partial_group is not None

    return partial_group


def chunks(list_, n):
    n = max(1, n)
    return (list[i:i + n] for i in range(0, len(list_), n))


def _is_gradient_clip_op(op):
    return op.desc.has_attr("op_namescope") \
        and op.desc.attr("op_namescope").startswith("/gradient_clip")


def _is_weight_decay_op(op):
    return op.desc.has_attr("op_namescope") \
        and op.desc.attr("op_namescope").startswith("/regularization")


# FIXME 
def _is_amp_applied(pass_context):
    for pass_obj in pass_context.passes:
        if isinstance(pass_obj, AMPBackwardPass):
            return True
    return False


def is_amp_related_param_grad_cast_op(block, op):
    if op.type != "cast":
        return False
    if not is_backward_op(op):
        return False
    assert (len(op.desc.input_arg_names()) == 1)
    assert (len(op.desc.output_arg_names()) == 1)
    input_name, output_name = op.desc.input_arg_names()[
        0], op.desc.output_arg_names()[0]
    input_var = block.var(input_name)
    output_var = block.var(output_name)
    if input_var.dtype != core.VarDesc.VarType.FP16 or \
        output_var.dtype != core.VarDesc.VarType.FP32:
        return False

    base_name = output_name[:output_name.find("@")]
    if not block.has_var(base_name):
        return False

    return block.var(base_name).is_parameter


class ShardingInfo(object):
    def __init__(self, group, rank, params):
        self.group = group
        self.params = params
        self.group_size = group.nranks
        self.global_rank = rank
        self.local_rank = group.ranks.index(self.global_rank)
        # rank in below mapping are local rank in this sharding group
        self.rank2params = shard_parameters(self.params, self.group_size)
        self.param2rank = dict()
        self._map_param_to_rank()
        self._param_names_in_local_shard = set()
        self._shard_params()

    def _map_param_to_rank(self):
        """
        mapping parameters to the rank which holds it.
        """
        for rank, params in self.rank2params.items():
            for param in params:
                self.param2rank[param] = rank

    def _shard_params(self):
        for p, rank in self.param2rank.items():
            if rank == self.local_rank:
                self._param_names_in_local_shard.add(p.name)
        assert len(self._param_names_in_local_shard) > 0

    def is_in_local_shard(self, param_name):
        return param_name in self._param_names_in_local_shard


def shard_parameters(params, group_size):

    # TODO(JZ-LIANG) support multiple partition methods
    # method1: greedy even but unorder
    # method2: roughly even with oreder

    mapping = {}
    for rank_ in range(group_size):
        mapping[rank_] = []
    sizes = [0] * group_size
    for param in params:
        rank = sizes.index(min(sizes))
        mapping[rank].append(param)
        numel = reduce(lambda x, y: x * y, param.shape)
        assert numel > 0, "param [{}] should larger than 0, but it is [{}]".format(
            param.name, numel)
        sizes[rank] += numel

    return mapping
