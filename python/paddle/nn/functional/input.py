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

from paddle import _C_ops

from ...common_ops_import import Variable
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.framework import in_dygraph_mode
from ...fluid.layer_helper import LayerHelper

__all__ = []


def one_hot(x, num_classes, name=None):
    """

    The operator converts each id in the input 'x' to an one-hot vector with a
    num_classes length. The value in the vector dimension corresponding to the id
    is 1, and the value in the remaining dimension is 0.

    The shape of output Tensor is generated by appending num_classes dimension
    behind the last dimension of the 'x' shape.

    .. code-block:: text

        Example 1:

        input:
            x.shape = [4]
            x.data = [1, 1, 3, 0]
            num_classes = 4

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.],
                        [1., 0., 0., 0.]]

        Example 2:

        input:
            x.shape = [4]
            x.data = [1, 1, 5, 0]
            num_classes = 4

        output: Throw an exception for Illegal value
            The second dimension in X is 5, which is greater than num_classes,
            so it throws an exception.


    Args:
        x(Tensor): Tensor with shape :math:`[N_1, N_2, ..., N_k]` ,
            which contains at least one dimension. The data type is int32 or int64.
        num_classes(int): An integer defining the num_classes of the one hot dimension. If input 'x'
            is word id, num_classes is generally the dictionary size.

    Returns:
        Tensor: The one-hot representations of 'x'. A Tensor with type float32.

    Examples:
        .. code-block:: python

            import paddle
            # Correspond to the first example above, where label.shape is 4 and one_hot_label.shape is [4, 4].
            label = paddle.to_tensor([1, 1, 3, 0], dtype='int64')
            # label.shape = [4]
            one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
            # one_hot_label.shape = [4, 4]
            # one_hot_label = [[0., 1., 0., 0.],
            #                  [0., 1., 0., 0.],
            #                  [0., 0., 0., 1.],
            #                  [1., 0., 0., 0.]]

    """

    if in_dygraph_mode():
        return _C_ops.one_hot(x, num_classes)
    else:
        check_variable_and_dtype(x, 'input', ['int32', 'int64'], 'one_hot_v2')
        helper = LayerHelper("one_hot_v2", **locals())

        one_hot_out = helper.create_variable_for_type_inference(dtype='float32')
        if not isinstance(num_classes, Variable):
            # user attribute
            inputs = {'X': x}
            attrs = {'depth': num_classes, 'allow_out_of_range': False}
        else:
            num_classes.stop_gradient = True
            inputs = {'X': x, 'depth_tensor': num_classes}
            attrs = {'allow_out_of_range': False}
        helper.append_op(
            type="one_hot_v2",
            inputs=inputs,
            attrs=attrs,
            outputs={'Out': one_hot_out},
            stop_gradient=True,
        )
        return one_hot_out


def embedding(x, weight, padding_idx=None, sparse=False, name=None):
    r"""
    Used to lookup embeddings vector of ids provided by :attr:`x` .

    The shape of output Tensor is generated by appending the last dimension of the input Tensor shape
    with embedding size.

    Note:
        The id in :attr:`x` must satisfy :math:`0 =< id < weight.shape[0]` ,
        otherwise the program will throw an exception and exit.

    .. code-block:: text

            x is a Tensor.
                padding_idx = -1
                x.data = [[1, 3], [2, 4], [4, 127]]
                x.shape = [3, 2]
                weight.shape = [128, 16]
            output is a Tensor:
                out.shape = [3, 2, 16]
                out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                            [0.345421456, 0.524563927, ..., 0.144534654]],
                            [[0.345249859, 0.124939536, ..., 0.194353745],
                            [0.945345345, 0.435394634, ..., 0.435345365]],
                            [[0.945345345, 0.435394634, ..., 0.435345365],
                            [0.0,         0.0,         ..., 0.0        ]]]  # padding data

            The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
            It will pad all-zero data when id is 127.

    Args:
        x(Tensor): A Tensor with type int32/int64, which contains the id information. The value of the input id should
            satisfy :math:`0<= id < weight.shape[0]` .
        weight (Tensor): The weight. A Tensor with shape of lookup table parameter. It should have two elements which
            indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
        sparse(bool, optional): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizers does not support sparse update,
            such as :ref:`api_paddle_optimizer_adadelta_Adadelta` , :ref:`api_paddle_optimizer_adamax_Adamax` , :ref:`api_paddle_optimizer_lamb_Lamb`.
            In these cases, sparse must be False. Default: False.
        padding_idx(int|long|None, optional): padding_idx needs to be in the interval [-weight.shape[0], weight.shape[0]).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`weight.shape[0] + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        name(str|None, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        Tensor: Embedding Tensor  mapped by x. The data type is the same as :attr:`weight`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn as nn

            x0 = paddle.arange(3, 6).reshape((3, 1)).astype(paddle.int64)
            w0 = paddle.full(shape=(10, 3), fill_value=2).astype(paddle.float32)

            # x.data = [[3], [4], [5]]
            # x.shape = [3, 1]
            x = paddle.to_tensor(x0, stop_gradient=False)

            # w.data = [[2. 2. 2.] ... [2. 2. 2.]]
            # w.shape = [10, 3]
            w = paddle.to_tensor(w0, stop_gradient=False)

            # emb.data = [[[2., 2., 2.]], [[2., 2., 2.]], [[2., 2., 2.]]]
            # emb.shape = [3, 1, 3]
            emb = nn.functional.embedding(
                    x=x, weight=w, sparse=True, name="embedding")

    """
    padding_idx = (
        -1
        if padding_idx is None
        else padding_idx
        if padding_idx >= 0
        else (weight.shape[0] + padding_idx)
    )

    if padding_idx >= weight.shape[0] or padding_idx < -weight.shape[0]:
        raise ValueError(
            "padding_idx must be within [-{}, {})".format(
                weight.shape[0], weight.shape[0]
            )
        )

    if in_dygraph_mode():
        return _C_ops.embedding(x, weight, padding_idx, sparse)
    else:
        helper = LayerHelper('embedding', **locals())
        dtype = helper.input_dtype(input_param_name='weight')

        check_variable_and_dtype(
            x,
            'input',
            ['uint8', 'int8', 'int16', 'int32', 'int64'],
            'embedding',
        )

        is_distributed = False
        remote_prefetch = sparse and (not is_distributed)

        tmp = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type='lookup_table_v2',
            inputs={'Ids': x, 'W': weight},
            outputs={'Out': tmp},
            attrs={
                'is_sparse': sparse,
                'is_distributed': is_distributed,
                'remote_prefetch': remote_prefetch,
                'padding_idx': padding_idx,
            },
        )
        return tmp


def embedding_bag(input, params, weight, mode, name=None):
    """
    Used to calculate the sum or average of the specified bag in the embeddings vector by : attr:'input' .
    Each bag contains several row indexes of embeddings.
    The shape of output Tensor is generated by appending the last dimension of 'params' shape behind the first
    dimension of the 'input' shape.
    Note:
        The id in :attr:'input' must satisfy :math: '0 <= id < params.shape[0]', and the shape of 'input' and 'weight' must be the same,
        otherwise the program will throw an exception and exit.
    .. code-block:: text
        x is a Tensor.
            mode = "sum"
            x.data = [[3, 3], [2, 3], [7, 6]]
            x.shape = [3,2]
            weight.shape = [3,2]
            params.shape = [10,3]
        output is a Tensor:
            out.shape = [3,3]
            out.data = [[ 7.24666166, -3.05312395, -6.47065353],
                        [ 2.96320534, -1.02136743, -3.56395578],
                        [-0.06210172, -2.78675747,  1.96624613]]

    Args:
        input(Tensor): A tensor with type int32/int64, which contains the id information. The shape is [bag_number, sequence_length],
                    The value of the input id should satisfy :math: `0 <= id < params.shape[0]` .
        params(Tensor): A tensor with shape of [num_embedding, embedding_dim] in which num_embedding indicates the size of the dictionary of embeddings
                    and embedding_dim indicates the size of each embedding vector.
        weight(Tensor): A tensor with the same shape of input. The variable can only be decalred when mode is set "sum". When mode is "mean", the value of
                    weight is set to 1 by default.
        mode(str): The string indicates calculation mode. mode can be set "sum" or "mean" for now.
        name(str|None, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and
                    None by default.
    Returns:
        Tensor: The calculation of embedding params according to 'input'. The data type is the same as 'params'.
    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn as nn
            # an EmbeddingBag module containing 10 tensors of size 3
            input = np.random.randint(low=0, high=10, size = (2,6)).astype(np.int64)
            weight = np.random.random((2,6)).astype(np.float32)
            params = np.random.random((10,3)).astype(np.float32)
            input = paddle.to_tensor(input, stop_gradient = False)
            weight = paddle.to_tensor(weight, stop_gradient = False)
            params = paddle.to_tensor(params, stop_gradient = False)
            # a batch of 2 samples of w indices each
            sum = nn.functional.embedding_bag(input, params, weight, mode='sum')
            # tensor([[ 0.7792,  0.3368, -0.0712],
            #    [-0.1101,  1.1775,  1.5315]])
    """

    if in_dygraph_mode():
        return _C_ops.embedding_bag(input, params, weight, mode)
    else:
        helper = LayerHelper('embedding_bag', **locals())
        dtype = helper.input_dtype(input_param_name='weight')

        check_variable_and_dtype(
            input,
            'input',
            ['uint8', 'int8', 'int16', 'int32', 'int64'],
            'embedding_bag',
        )

        # is_distributed = False

        tmp = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type='embedding_bag',
            inputs={
                'input': input,
                'params': params,
                'weight': weight,
            },
            outputs={'out': tmp},
            attrs={'mode': mode},
        )
        return tmp
