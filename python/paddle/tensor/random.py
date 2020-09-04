#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define random functions  

from ..fluid import core
from ..fluid.framework import in_dygraph_mode, Variable, convert_np_dtype_to_dtype_
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, check_shape
from ..fluid.layers import utils
import paddle

from ..fluid.io import shuffle  #DEFINE_ALIAS

__all__ = [
    'bernoulli',
    'standard_normal',
    'normal',
    'uniform',
    'shuffle',
    'randn',
    'rand',
    'randint',
    'randperm',
]


def bernoulli(x, name=None):
    """

    This OP returns a Tensor filled with random binary(0 or 1) number from a Bernoulli distribution.
    The input ``x`` is a tensor with probabilities for generating the random binary number.
    Each element in ``x`` should be in [0, 1], and the out is generated by:
    
    .. math::

        out_i ~ Bernoulli (x_i)

    Args:
        x(Tensor):  A tensor with probabilities for generating the random binary number. The data type 
            should be float32, float64.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns: 
        Tensor: A Tensor filled with random binary number with the same shape and dtype as ``x``.

    Examples:
        .. code-block:: python

        import paddle

        paddle.disable_static()

        x = paddle.rand([2, 3])
        print(x.numpy())
        # [[0.11272584 0.3890902  0.7730957 ]
        # [0.10351662 0.8510418  0.63806665]]

        out = paddle.bernoulli(x)
        print(out.numpy())
        # [[0. 0. 1.]
        # [0. 0. 1.]]

    """

    if in_dygraph_mode():
        return core.ops.bernoulli(x)

    check_variable_and_dtype(x, "x", ["float32", "float64"], "bernoulli")

    helper = LayerHelper("randint", **locals())
    out = helper.create_variable_for_type_inference(
        dtype=x.dtype)  # maybe set out to int32 ? 
    helper.append_op(
        type='bernoulli', inputs={"X": x}, outputs={'Out': out}, attrs={})
    return out


def gaussian(shape, mean=0.0, std=1.0, dtype=None, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a Gaussian
    distribution, with ``shape`` and ``dtype``.

    Args:
        shape (list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        mean (float|int, optional): Mean of the output tensor, default is 0.0.
        std (float|int, optional): Standard deviation of the output tensor, default
            is 1.0.
        seed (int, optional): Random seed of generator.
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a Gaussian
        distribution, with ``shape`` and ``dtype``. 
    """
    op_type_for_check = 'gaussian/standard_normal/randn/normal'
    seed = 0

    if dtype is None:
        dtype = paddle.framework.get_default_dtype()
        if dtype not in ['float32', 'float64']:
            raise TypeError(
                "{} only supports [float32, float64], but the default dtype is {}"
                .format(op_type_for_check, dtype))
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        return core.ops.gaussian_random('shape', shape, 'mean',
                                        float(mean), 'std',
                                        float(std), 'seed', seed, 'dtype',
                                        dtype)

    check_shape(shape, op_type_for_check)
    check_dtype(dtype, 'dtype', ['float32', 'float64'], op_type_for_check)

    inputs = {}
    attrs = {
        'mean': mean,
        'std': std,
        'seed': seed,
        'dtype': dtype,
        'use_mkldnn': False
    }
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type=op_type_for_check)

    helper = LayerHelper('gaussian', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='gaussian_random',
        inputs=inputs,
        outputs={'Out': out},
        attrs=attrs)
    out.stop_gradient = True
    return out


def standard_normal(shape, dtype=None, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a standard
    normal distribution with mean 0 and standard deviation 1, with ``shape``
    and ``dtype``.

    Args:
        shape (list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a standard
        normal distribution with mean 0 and standard deviation 1, with
        ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()

            # example 1: attr shape is a list which doesn't contain Tensor.
            out1 = paddle.standard_normal(shape=[2, 3])
            # [[-2.923464  ,  0.11934398, -0.51249987],  # random
            #  [ 0.39632758,  0.08177969,  0.2692008 ]]  # random

            # example 2: attr shape is a list which contains Tensor.
            dim1 = paddle.full([1], 2, "int64")
            dim2 = paddle.full([1], 3, "int32")
            out2 = paddle.standard_normal(shape=[dim1, dim2, 2])
            # [[[-2.8852394 , -0.25898588],  # random
            #   [-0.47420555,  0.17683524],  # random
            #   [-0.7989969 ,  0.00754541]],  # random
            #  [[ 0.85201347,  0.32320443],  # random
            #   [ 1.1399018 ,  0.48336947],  # random
            #   [ 0.8086993 ,  0.6868893 ]]]  # random

            # example 3: attr shape is a Tensor, the data type must be int64 or int32.
            shape_tensor = paddle.to_tensor([2, 3])
            result_3 = paddle.standard_normal(shape_tensor)
            # [[-2.878077 ,  0.17099959,  0.05111201]  # random
            #  [-0.3761474, -1.044801  ,  1.1870178 ]]  # random

    """
    return gaussian(shape=shape, mean=0.0, std=1.0, dtype=dtype, name=name)


randn = standard_normal


def normal(mean=0.0, std=1.0, shape=None, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a normal
    distribution with ``mean`` and ``std`` (standard deviation) .

    If ``mean`` is a Tensor, the output Tensor has the same shape and data type as ``mean``.
    If ``mean`` is not a Tensor and ``std`` is a Tensor, the output Tensor has the same shape and data type as ``std``.
    If ``mean`` and ``std`` are not a Tensor, the output Tensor has the same shape as ``shape``, with data type float32.

    If ``mean`` and ``std`` are Tensor, the num of elements of ``mean`` and ``std`` should be the same.

    Args:
        mean (float|Tensor, optional): The mean of the output Tensor's normal distribution.
            If ``mean`` is float, all elements of the output Tensor shared the same mean.
            If ``mean`` is a Tensor(data type supports float32, float64), it has per-element means.
            Default is 0.0
        std (float|Tensor, optional): The  standard deviation of the output Tensor's normal distribution.
            If ``std`` is float, all elements of the output Tensor shared the same standard deviation.
            If ``std`` is a Tensor(data type supports float32, float64), it has per-element standard deviations.
            Defaule is 1.0
        shape (list|tuple|Tensor, optional): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64). If ``mean`` or ``std`` is a Tensor, the shape of the output
            Tensor is the same as ``mean`` or ``std`` , attr ``shape`` is ignored.
            Default is None
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor filled with random values sampled from a normal distribution with ``mean`` and ``std`` .

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()

            out1 = paddle.normal(shape=[2, 3])
            # [[ 0.17501129  0.32364586  1.561118  ]  # random
            #  [-1.7232178   1.1545963  -0.76156676]]  # random

            mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
            out2 = paddle.normal(mean=mean_tensor)
            # [ 0.18644847 -1.19434458  3.93694787]  # random

            std_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
            out3 = paddle.normal(mean=mean_tensor, std=std_tensor)
            # [1.00780561 3.78457445 5.81058198]  # random

    """
    if not in_dygraph_mode():
        check_type(mean, 'mean', (int, float, Variable), 'normal')
        check_type(std, 'std', (int, float, Variable), 'normal')
        if isinstance(mean, Variable):
            check_dtype(
                mean.dtype, 'mean', ['float32', 'float64'], 'normal',
                "If mean is Tensor, it's data type only support float32, float64."
            )
        if isinstance(std, Variable):
            check_dtype(
                std.dtype, 'std', ['float32', 'float64'], 'normal',
                "If std is Tensor, it's data type only support float32, float64."
            )
        if shape is not None:
            check_shape(shape, 'normal')

    if isinstance(mean, Variable):
        if isinstance(std, Variable):
            if std.dtype != mean.dtype:
                std = paddle.cast(std, mean.dtype)
            mean_shape = paddle.shape(mean)
            std = paddle.reshape(std, mean_shape)
        else:
            std = float(std)
        out = standard_normal(paddle.shape(mean), mean.dtype, name)
    elif isinstance(std, Variable):
        mean = float(mean)
        out = standard_normal(paddle.shape(std), std.dtype, name)
    else:
        return gaussian(shape=shape, mean=mean, std=std, name=name)

    out = out * std + mean
    if not in_dygraph_mode():
        out.stop_grediant = True
    return out


def uniform(shape, dtype=None, min=-1.0, max=1.0, seed=0, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Examples:
    ::
        Input:
          shape = [1, 2]
        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype(str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        min(float|int, optional): The lower bound on the range of random values
            to generate, ``min`` is included in the range. Default is -1.0.
        max(float|int, optional): The upper bound on the range of random values
            to generate, ``max`` is excluded in the range. Default is 1.0.
        seed(int, optional): Random seed used for generating samples. 0 means
            use a seed generated by the system. Note that if seed is not 0,
            this operator will always generate the same random numbers every
            time. Default is 0.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Raises:
        TypeError: If ``shape`` is not list, tuple, Tensor.
        TypeError: If ``dtype`` is not float32, float64.

    Examples:
        .. code-block:: python
            
            import paddle

            paddle.disable_static()

            # example 1:
            # attr shape is a list which doesn't contain Tensor.
            result_1 = paddle.tensor.random.uniform(shape=[3, 4])
            # [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357],
            #  [-0.34646994, -0.45116323, -0.09902662, -0.11397249],
            #  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]]

            # example 2:
            # attr shape is a list which contains Tensor.
            dim_1 = paddle.fill_constant([1], "int64", 2)
            dim_2 = paddle.fill_constant([1], "int32", 3)
            result_2 = paddle.tensor.random.uniform(shape=[dim_1, dim_2])
            # [[-0.9951253,   0.30757582, 0.9899647 ],
            #  [ 0.5864527,   0.6607096,  -0.8886161 ]]

            # example 3:
            # attr shape is a Tensor, the data type must be int64 or int32.
            shape_tensor = paddle.to_tensor([2, 3])
            result_3 = paddle.tensor.random.uniform(shape_tensor)
            # if shape_tensor's value is [2, 3]
            # result_3 is:
            # [[-0.8517412,  -0.4006908,   0.2551912 ],
            #  [ 0.3364414,   0.36278176, -0.16085452]]


    """
    if dtype is None:
        dtype = paddle.framework.get_default_dtype()
        if dtype not in ['float32', 'float64']:
            raise TypeError(
                "uniform/rand only supports [float32, float64], but the default dtype is {}".
                format(dtype))

    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        return core.ops.uniform_random('shape', shape, 'min',
                                       float(min), 'max',
                                       float(max), 'seed', seed, 'dtype', dtype)

    check_type(shape, 'shape', (list, tuple, Variable), 'uniform/rand')
    check_dtype(dtype, 'dtype', ('float32', 'float64'), 'uniform/rand')

    inputs = dict()
    attrs = {'seed': seed, 'min': min, 'max': max, 'dtype': dtype}
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='uniform/rand')

    helper = LayerHelper("uniform", **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="uniform_random", inputs=inputs, attrs=attrs,
        outputs={"Out": out})
    return out


def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    """
    This OP returns a Tensor filled with random integers from a discrete uniform
    distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.
    If ``high`` is None (the default), the range is [0, ``low``).

    Args:
        low (int): The lower bound on the range of random values to generate.
            The ``low`` is included in the range. If ``high`` is None, the
            range is [0, ``low``). Default is 0.
        high (int, optional): The upper bound on the range of random values to
            generate, the ``high`` is excluded in the range. Default is None
            (see above for behavior if high = None). Default is None.
        shape (list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64). Default is [1].
        dtype (str|np.dtype, optional): The data type of the
            output tensor. Supported data types: int32, int64. If ``dytpe``
            is None, the data type is int64. Default is None.
        name (str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: 
        Tensor: A Tensor filled with random integers from a discrete uniform
        distribution in the range [``low``, ``high``), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()

            # example 1:
            # attr shape is a list which doesn't contain Tensor.
            out1 = paddle.randint(low=-5, high=5, shape=[3])
            # [0, -3, 2]  # random

            # example 2:
            # attr shape is a list which contains Tensor.
            dim1 = paddle.full([1], 2, "int64")
            dim2 = paddle.full([1], 3, "int32")
            out2 = paddle.randint(low=-5, high=5, shape=[dim1, dim2], dtype="int32")
            # [[0, -1, -3],  # random
            #  [4, -2,  0]]  # random

            # example 3:
            # attr shape is a Tensor
            shape_tensor = paddle.to_tensor(3)
            result_3 = paddle.randint(low=-5, high=5, shape=shape_tensor)
            # [-2, 2, 3]  # random

            # example 4:
            # data type is int32
            out4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')
            # [-5, 4, -4]  # random

            # example 5:
            # Input only one parameter
            # low=0, high=10, shape=[1], dtype='int64'
            out5 = paddle.randint(10)
            # [7]  # random

    """
    if high is None:
        if low <= 0:
            raise ValueError(
                "If high is None, low must be greater than 0, but received low = {0}.".
                format(low))
        high = low
        low = 0
    if dtype is None:
        dtype = 'int64'
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        return core.ops.randint('shape', shape, 'low', low, 'high', high,
                                'seed', 0, 'dtype', dtype)

    check_shape(shape, 'randint')
    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')
    if low >= high:
        raise ValueError(
            "randint's low must less then high, but received low = {0}, "
            "high = {1}".format(low, high))

    inputs = dict()
    attrs = {'low': low, 'high': high, 'seed': 0, 'dtype': dtype}
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='randint')

    helper = LayerHelper("randint", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def randperm(n, dtype="int64", name=None):
    """
    This OP returns a 1-D Tensor filled with random permutation values from 0
    to n-1, with ``dtype``.

    Args:
        n (int): The upper bound (exclusive), and it should be greater than 0.
        dtype (str|np.dtype, optional): The data type of
            the output Tensor. Supported data types: int32, int64, float32,
            float64. Default is int64.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A 1-D Tensor filled with random permutation values from 0
        to n-1, with ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()

            out1 = paddle.randperm(5)
            # [4, 1, 2, 3, 0]  # random

            out2 = paddle.randperm(7, 'int32')
            # [1, 6, 2, 0, 4, 3, 5]  # random
 
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        return core.ops.randperm('n', n, 'seed', 0, 'dtype', dtype)

    if n < 1:
        raise ValueError("The input n should be greater than 0 in randperm op.")
    check_dtype(dtype, 'dtype', ['int64', 'int32', 'float32', 'float64'],
                'randperm')

    helper = LayerHelper("randperm", **locals())
    out = helper.create_variable_for_type_inference(dtype)
    attrs = {'n': n, 'dtype': dtype, 'seed': 0}
    helper.append_op(
        type='randperm', inputs={}, outputs={'Out': out}, attrs=attrs)
    out.stop_gradient = True
    return out


def rand(shape, dtype=None, name=None):
    """
    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Args:
        shape (list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype (str|np.dtype, optional): The data type of the output Tensor.
            Supported data types: float32, float64.
            Default is None, use global default dtype (see ``get_default_dtype``
            for details).
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [0, 1), with ``shape`` and ``dtype``.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            # example 1: attr shape is a list which doesn't contain Tensor.
            out1 = paddle.rand(shape=[2, 3])
            # [[0.451152  , 0.55825245, 0.403311  ],  # random
            #  [0.22550228, 0.22106001, 0.7877319 ]]  # random

            # example 2: attr shape is a list which contains Tensor.
            dim1 = paddle.full([1], 2, "int64")
            dim2 = paddle.full([1], 3, "int32")
            out2 = paddle.rand(shape=[dim1, dim2, 2])
            # [[[0.8879919 , 0.25788337],  # random
            #   [0.28826773, 0.9712097 ],  # random
            #   [0.26438272, 0.01796806]],  # random
            #  [[0.33633623, 0.28654453],  # random
            #   [0.79109055, 0.7305809 ],  # random
            #   [0.870881  , 0.2984597 ]]]  # random

            # example 3: attr shape is a Tensor, the data type must be int64 or int32.
            shape_tensor = paddle.to_tensor([2, 3])
            result_3 = paddle.rand(shape_tensor)

            # [[0.22920267, 0.841956  , 0.05981819],  # random
            #  [0.4836288 , 0.24573246, 0.7516129 ]]  # random

    """
    return uniform(shape, dtype, min=0.0, max=1.0, name=name)
