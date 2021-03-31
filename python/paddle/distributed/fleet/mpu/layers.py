#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.dygraph.layers import Layer
from .random import get_rng_state_tracker
from paddle.nn import functional as F

__all__ = [
    'ParallelLinear', 'ColumnParallelLinear', 'RowParallelLinear',
    'VocabParallelEmbedding'
]


def _get_model_parallel_messgae():
    import paddle.distributed.fleet as fleet
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    local_rank = hcg.get_model_parallel_rank()
    src_global_rank = hcg.get_model_parallel_group_src_rank()

    return model_parallel_group, world_size, local_rank, src_global_rank


def _get_data_parallel_messgae():
    import paddle.distributed.fleet as fleet
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_data_parallel_group()
    world_size = hcg.get_data_parallel_world_size()
    local_rank = hcg.get_data_parallel_rank()
    src_global_rank = hcg.get_data_parallel_group_src_rank()

    return model_parallel_group, world_size, local_rank, src_global_rank


class ColumnParallelLinear(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 has_bias=None,
                 gather_output=True,
                 name=None):
        super(ColumnParallelLinear, self).__init__()

        self.model_parallel_group, self.world_size, _, _ = _get_model_parallel_messgae(
        )
        self.name = name
        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(out_features,
                                                            self.world_size))
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        with get_rng_state_tracker().rng_state():
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype)
            self.weight.is_distributed = True

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype)
            self.bias.is_distributed = True
        else:
            self.bias = None

    def forward(self, x):
        linear_out = F.linear(x, self.weight, self.bias, name=self.name)
        if self.gather_output:
            output = []
            paddle.distributed.all_gather(
                output, linear_out, group=self.model_parallel_group)
            linear_out = paddle.concat(output, axis=len(linear_out.shape) - 1)

        return linear_out


class RowParallelLinear(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 has_bias=True,
                 input_is_parallel=False,
                 name=None):
        super(RowParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self.name = name
        self.model_parallel_group, self.world_size, self.rank, _ = _get_model_parallel_messgae(
        )

        assert in_features % self.world_size == 0, (
            "Number of row of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(in_features,
                                                            self.world_size))

        self.input_size_per_partition = in_features // self.world_size

        with get_rng_state_tracker().rng_state():
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype)
            self.weight.is_distributed = True

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype)
        else:
            self.bias = None

    def forward(self, x):
        if self.input_is_parallel:
            input_parallel = x
        else:
            # split last dim
            last_dim = len(x.shape) - 1
            input_list = paddle.split(
                x, num_or_sections=self.world_size, axis=last_dim)
            input_parallel = input_list[self.rank]

        output_parallel = F.linear(input_parallel, self.weight, name=self.name)
        output_ = paddle.distributed.all_reduce(
            output_parallel, group=self.model_parallel_group)
        output = output_ + self.bias if self.bias is not None else output_
        return output


class ParallelLinear(Layer):
    def __init__(self,
                 size,
                 axis=0,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super(ParallelLinear, self).__init__()

        self.model_parallel_group, self.world_size, _, _ = _get_model_parallel_messgae(
        )

        assert num_partitions == self.world_size, \
            "num_partitions must be equal to the model parallel world size"

        if axis == 0:
            assert size[0] % num_partitions == 0, (
                "Number of rows of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[0],
                                                           num_partitions))
            per_part_size = size[0] // num_partitions
            linear_size = (per_part_size, size[1])

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                "Number of column of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[1],
                                                           num_partitions))
            per_part_size = size[1] // num_partitions
            linear_size = (size[0], per_part_size)
        else:
            raise ValueError("The value of axis must be 0 or 1, but the value "
                             "given is {}.".format(axis))

        self.linear = paddle.nn.Linear(
            linear_size[0],
            linear_size[1],
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)
        self.linear_size = linear_size
        self.axis = axis
        self.gather_out = gather_out

    def forward(self, x):
        linear_out = self.linear(x)
        if self.gather_out:
            if self.axis == 0:
                linear_out = paddle.distributed.all_reduce(
                    linear_out, group=self.model_parallel_group)
            else:
                output = []
                paddle.distributed.all_gather(
                    output, linear_out, group=self.model_parallel_group)
                linear_out = paddle.concat(
                    output, axis=len(linear_out.shape) - 1)
        return linear_out


class VocabParallelEmbedding(Layer):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 weight_attr=None,
                 name=None):
        super(VocabParallelEmbedding, self).__init__()
        self.model_parallel_group, self.world_size, self.rank, _ = _get_model_parallel_messgae(
        )
        self.origin_num_embeddings = num_embeddings

        per_part_size = (
            num_embeddings + self.world_size - 1) // self.world_size
        last_part_size = num_embeddings - per_part_size * (self.world_size - 1)
        if self.rank == self.world_size - 1:
            per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index
        self.per_part_size = per_part_size

        with get_rng_state_tracker().rng_state():
            self.embedding = paddle.nn.Embedding(
                per_part_size,
                embedding_dim,
                padding_idx=per_part_size - 1,
                sparse=False,
                weight_attr=weight_attr,
                name=name)
            self.embedding.weight.is_distributed = True

    def forward(self, x):
        origin_input_shape = x.shape
        if len(origin_input_shape) == 2:
            x = paddle.unsqueeze(x, axis=-1)
        else:
            assert origin_input_shape[-1] == 1, (
                "The last dimension size of x must be 1.")
        x_shard = paddle.shard_index(x, self.origin_num_embeddings,
                                     self.world_size, self.rank,
                                     self.per_part_size - 1)
        if len(origin_input_shape) == 2:
            x_shard = paddle.squeeze(x_shard, axis=-1)
        emb_out_ = self.embedding(x_shard)
        emb_out = paddle.distributed.all_reduce(
            emb_out_, group=self.model_parallel_group)
        return emb_out
