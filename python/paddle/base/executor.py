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

import copy
import logging
import os
import sys
import warnings
from functools import lru_cache

import numpy as np

from ..pir import OpResult, translate_to_new_ir
from . import compiler, core, framework, get_flags, set_flags, unique_name
from .data_feeder import convert_dtype
from .framework import (
    Operator,
    Program,
    Variable,
    _apply_pass,
    convert_np_dtype_to_dtype_,
    default_main_program,
    in_pir_mode,
    paddle_type_to_proto_type,
)
from .incubate.checkpoint import auto_checkpoint as acp
from .trainer_factory import FetchHandlerMonitor, TrainerFactory
from .wrapped_decorator import signature_safe_contextmanager

__all__ = []

g_scope = core.Scope()
InferNativeConfig = core.NativeConfig
InferAnalysisConfig = core.AnalysisConfig


def global_scope():
    """
    :api_attr: Static Graph

    Get the global/default scope instance. There are a lot of APIs use
    :code:`global_scope` as its default value, e.g., :code:`Executor.run`

    Returns:
        Scope: The global/default scope instance.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy

            >>> paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
            >>> numpy.array(paddle.static.global_scope().find_var("data").get_tensor())
    """
    return g_scope


def _switch_scope(scope):
    global g_scope
    ex = g_scope
    g_scope = scope
    return ex


@signature_safe_contextmanager
def scope_guard(scope):
    """

    This function switches scope through python `with` statement.
    Scope records the mapping between variable names and variables ( :ref:`api_guide_Variable` ),
    similar to brackets in programming languages.
    If this function is not invoked, all variables and variable names are recorded in the default global scope.
    When users need to create variables with the same name,
    they need to switch scopes through this function
    if they do not want the mapping of variables with the same name to be overwritten.
    After switching through the `with` statement,
    all variables created in the `with` block will be assigned to a new scope.

    Parameters:
        scope: The new scope.

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import numpy
            >>> paddle.enable_static()

            >>> new_scope = paddle.static.Scope()
            >>> with paddle.static.scope_guard(new_scope):
            ...         paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
            >>> numpy.array(new_scope.find_var("data").get_tensor())
            array([[1., 1.],
                   [1., 1.]])
    """

    ex = _switch_scope(scope)
    try:
        yield
    finally:
        _switch_scope(ex)


def as_numpy(tensor, copy=False):
    """
    Convert a Tensor to a numpy.ndarray, its only support Tensor without LoD information.
    For higher dimensional sequence data, please use LoDTensor directly.

    Examples:
        .. code-block:: python

            >>> import paddle.base as base
            >>> import numpy

            >>> new_scope = base.Scope()
            >>> with base.scope_guard(new_scope):
            ...     base.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), base.CPUPlace())
            >>> tensor = new_scope.find_var("data").get_tensor()
            >>> base.executor.as_numpy(tensor) # or numpy.array(new_scope.find_var("data").get_tensor())

    Args:
       tensor(Variable): a instance of Tensor
       copy(bool, optional): Whether to use deep copy.

    Returns:
        numpy.ndarray
    """
    if isinstance(tensor, core.LoDTensorArray):
        return [as_numpy(t, copy) for t in tensor]
    if isinstance(tensor, list):
        return [as_numpy(t, copy) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    if len(lod) > 0:
        raise RuntimeError(
            "Some of your fetched tensors hold LoD information. \
            They can not be completely cast to Python ndarray. \
            Please set the parameter 'return_numpy' as 'False' to \
            return LoDTensor itself directly."
        )
    if tensor._is_initialized():
        if copy:
            return np.array(tensor)
        else:
            return np.asarray(tensor)
    else:
        return None


def dtype_is_compatible_with(first, second):
    """
    Returns True if the first dtype can be compatible the second one.
    Currently, we require the two dtype's have to be same.

    Args:
        dtype (np.dtype|VarType|str): The type of data: float32, int64, etc.

    Returns:
        True if the two types are same.
    """
    if not isinstance(first, core.VarDesc.VarType):
        first = convert_np_dtype_to_dtype_(first)
    if not isinstance(second, core.VarDesc.VarType):
        second = convert_np_dtype_to_dtype_(second)
    return first == second


def dimension_is_compatible_with(first, second):
    """
    Returns True if the two dimensions are compatible.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimensions are same.
    3. For negative number or 'None' in a dimension, it means unknown so it
       is compatible with any number.

    Args:
        first (list/tuple): integers representing shape. "None" or negative
            number means unknown.
        second (list/tuple): integers representing shape. "None" or negative
            number means unknown.

    Returns:
        True if the two dimensions are compatible.
    """

    dim_len = len(first)
    if dim_len != len(second):
        return False

    for i in range(dim_len):
        if first[i] is None or first[i] < 0:
            continue
        if second[i] is None or second[i] < 0:
            continue
        if first[i] != second[i]:
            return False

    return True


def check_feed_shape_type(var, feed, num_places=1):
    """
    Returns True if the variable doesn't require feed check or it is compatible
    with the shape and have same dtype as the fed value.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimensions are same.
    3. For negative number or 'None' in a dimension, it means unknown so it
       is compatible with any number.

    Args:
        var (Variable): the Variable object
        feed (LoDTensor): the fed value, which must be a LoDTensor
        num_places: an integer value indicating the number of places.
            ParallelExecutor will divide data into devices (CPU/GPU) evenly.
    Returns:
        True if the shape and dtype of variable is compatible with the feed value
    Raises:
        ValueError: if the shape or dtype of the variable is not compatible with
            the feed value
    """
    if var.desc.need_check_feed():
        diff_shape = core.diff_tensor_shape(feed, var.desc, num_places)
        if diff_shape is not None:
            raise ValueError(
                'The fed Variable %r should have dimensions = %d, shape = '
                '%r, but received fed shape %r on each device'
                % (var.name, len(var.shape), var.shape, diff_shape)
            )
        if not dtype_is_compatible_with(feed._dtype(), var.dtype):
            var_dtype_format = (
                convert_dtype(var.dtype)
                if isinstance(var.dtype, core.VarDesc.VarType)
                else var.dtype
            )
            feed_dtype_format = (
                convert_dtype(feed._dtype())
                if isinstance(feed._dtype(), core.VarDesc.VarType)
                else feed._dtype()
            )
            raise ValueError(
                'The data type of fed Variable {!r} must be {!r}, but received {!r}'.format(
                    var.name, var_dtype_format, feed_dtype_format
                )
            )
    return True


def pir_check_feed_shape_type(feed, name, target_shape, dtype, num_places=1):
    """
    Returns True if the variable doesn't require feed check or it is compatible
    with the shape and have same dtype as the fed value.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimensions are same.
    3. For negative number or 'None' in a dimension, it means unknown so it
       is compatible with any number.

    Args:
        feed (LoDTensor): the fed value, which must be a LoDTensor
        name (str): name of the variable
        target_shape (list): the shape that will be compared with feed
        dtype (core.VarDesc.VarType): the dtype that will be compared with feed
        num_places: an integer value indicating the number of places.
            ParallelExecutor will divide data into devices (CPU/GPU) evenly.
    Returns:
        True if the shape and dtype of variable is compatible with the feed value
    Raises:
        ValueError: if the shape or dtype of the variable is not compatible with
            the feed value
    """
    diff_shape = core.diff_tensor_shape(feed, target_shape, num_places)
    if diff_shape is not None:
        raise ValueError(
            'The fed Variable %r should have dimensions = %d, shape = '
            '%r, but received fed shape %r on each device'
            % (name, len(target_shape), target_shape, diff_shape)
        )
    if not dtype_is_compatible_with(feed._dtype(), dtype):
        var_dtype_format = (
            convert_dtype(dtype)
            if isinstance(dtype, core.VarDesc.VarType)
            else dtype
        )
        feed_dtype_format = (
            convert_dtype(feed._dtype())
            if isinstance(feed._dtype(), core.VarDesc.VarType)
            else feed._dtype()
        )
        raise ValueError(
            'The data type of fed Variable {!r} must be {!r}, but received {!r}'.format(
                name, var_dtype_format, feed_dtype_format
            )
        )
    return True


def has_feed_operators(block, feed_targets, feed_holder_name):
    """Check whether the block already has feed operators.

    Return false if the block does not have any feed operators.
    If some feed operators have been prepended to the block, check that
    the info contained in these feed operators matches the feed_targets
    and feed_holder_name. Raise exception when any mismatch is found.
    Return true when the block has feed operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        feed_targets: a dictionary of {feed_target_name: feed_target_data}
        feed_holder_name: the name of the variable that holds the data of
            all feed targets. The type of this feed_holder variable is
            FEED_MINIBATCH, which is essentially vector<LoDTensor>.

    Returns:
        A boolean value that indicates whether a block has feed operators
        that match the info contained in feed_targets and feed_holder_name.
    """

    feed_count = 0
    for op in block.ops:
        if op.desc.type() == 'feed':
            feed_count += 1
            assert op.desc.input('X')[0] == feed_holder_name
            feed_target_name = op.desc.output('Out')[0]
            if feed_target_name not in feed_targets:
                raise Exception(
                    f"'feed_targets' does not have {feed_target_name} variable"
                )
        else:
            break
    if feed_count > 0 and feed_count != len(feed_targets):
        raise Exception(
            "Feed operators in program desc do not match 'feed_targets'"
        )
    return feed_count > 0


def has_fetch_operators(
    block, fetch_targets, fetch_holder_name, fetch_op='fetch'
):
    """Check whether the block already has fetch operators.

    Return false if the block does not have any fetch operators.
    If some fetch operators have been appended to the block, check that
    the info contained in these fetch operators matches the fetch_targets
    and fetch_holder_name. Raise exception when any mismatch is found.
    Return true when the block has fetch operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        fetch_targets: a dictionary of {fetch_target_name: fetch_target_data}
        fetch_holder_name: the name of the variable that holds the data of
            all fetch targets. The type of this fetch_holder variable is
            FETCH_LIST, which is essentially vector<LoDTensor>.
        fetch_op: the operator name of fetch

    Return:
        A boolean value that indicates whether a block has fetch operators
        that match the info contained in fetch_targets and fetch_holder_name.
    """

    fetch_count = 0
    for op in block.ops:
        if op.desc.type() == fetch_op:
            fetch_count += 1
            assert op.desc.output('Out')[0] == fetch_holder_name
            fetch_target_name = op.desc.input('X')[0]
            if fetch_target_name not in [
                var.desc.name() for var in fetch_targets
            ]:
                raise Exception(
                    f"'fetch_targets' does not have {fetch_target_name} variable"
                )
            idx = op.desc.attr('col')
            assert fetch_target_name == fetch_targets[idx].desc.name()
    if fetch_count > 0 and fetch_count != len(fetch_targets):
        raise Exception(
            "Fetch operators in program desc do not match 'fetch_targets'"
        )
    return fetch_count > 0


def has_fetch_operations(
    block, fetch_targets, fetch_holder_name, fetch_op='pd_op.fetch'
):
    """Check whether the block already has fetch operation.

    Return false if the block does not have any fetch operation.
    If some fetch operation have been appended to the block, check that
    the info contained in these fetch operation matches the fetch_targets.
    Raise exception when any mismatch is found.
    Return true when the block has fetch operation with matching info.

    Args:
        block: a block instance (typically global block of a program)
        fetch_targets: a list of fetch_target_data
        fetch_op: the operator name of fetch

    Return:
        A boolean value that indicates whether a block has fetch operators
        that match the info contained in fetch_targets.
    """

    fetch_count = 0
    for op in block.ops:
        if op.name() == fetch_op:
            fetch_count += 1
            if op.operand_source(0) not in fetch_targets:
                raise Exception(
                    "There is a fetch op in Program which will fetch variable that is not belong to fetch_targets."
                )

    if fetch_count > 0 and fetch_count != len(fetch_targets):
        raise Exception(
            "Fetch operations in program do not match 'fetch_targets'"
        )
    return fetch_count > 0


def _add_feed_fetch_ops(
    program, feed, fetch_list, feed_var_name, fetch_var_name, use_fetch_v2=False
):
    tmp_program = program.clone()

    global_block = tmp_program.global_block()

    if feed_var_name in global_block.vars:
        feed_var = global_block.var(feed_var_name)
    else:
        feed_var = global_block.create_var(
            name=feed_var_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True,
        )

    if fetch_var_name in global_block.vars:
        fetch_var = global_block.var(fetch_var_name)
    else:
        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True,
        )

    # prepend feed operators
    if not has_feed_operators(global_block, feed, feed_var_name):
        for i, name in enumerate(feed):
            if global_block.has_var(name):
                out = global_block.var(name)
                global_block._prepend_op(
                    type='feed',
                    inputs={'X': [feed_var]},
                    outputs={'Out': [out]},
                    attrs={'col': i},
                )
            else:
                warnings.warn(
                    "The variable %s is not found in program. It is not declared or is pruned."
                    % name
                )

    if use_fetch_v2:
        fetch_op = 'fetch_v2'
    else:
        fetch_op = 'fetch'

    # append fetch_operators
    if not has_fetch_operators(
        global_block, fetch_list, fetch_var_name, fetch_op
    ):
        for i, var in enumerate(fetch_list):
            assert isinstance(
                var, (Variable, str)
            ), f"Wrong type for fetch_list[{i}]: {type(var)}"
            global_block.append_op(
                type=fetch_op,
                inputs={'X': [var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i},
            )

    return tmp_program


def _add_pir_fetch_ops(program, fetch_list, fetch_var_name):
    import paddle

    global_block = program.global_block()
    fetch_op = "pd_op.fetch"
    if not has_fetch_operations(
        global_block, fetch_list, fetch_var_name, fetch_op
    ):
        with paddle.static.program_guard(program):
            for i, fetch_input in enumerate(fetch_list):
                assert isinstance(
                    fetch_input, OpResult
                ), f"Wrong type for fetch_list[{i}]: {type(fetch_input)}"
                paddle._pir_ops.fetch(fetch_input, fetch_var_name + str(i), i)


def _merge_tensors(tensor, micro_batch_num):
    if micro_batch_num <= 1:
        return tensor
    assert len(tensor) % micro_batch_num == 0
    chunk_tensor = [
        tensor[i : i + micro_batch_num]
        for i in range(0, len(tensor), micro_batch_num)
    ]
    return [np.array(chunk) for chunk in chunk_tensor]


def _apply_inplace_addto_pass(
    program, enable_inplace, enable_addto, skip_var_names
):
    use_cuda = True if core.is_compiled_with_cuda() else False

    attrs = {"use_cuda": use_cuda, "mem_opt_skip_vars": skip_var_names}
    attr_types = {"use_cuda": "bool", "mem_opt_skip_vars": "list[str]"}

    empty_startup_program = Program()
    if enable_inplace:
        pass_name = "buffer_shared_inplace_pass"
        _apply_pass(
            program, empty_startup_program, pass_name, attrs, attr_types
        )
    if enable_addto and use_cuda:
        pass_name = "inplace_addto_op_pass"
        _apply_pass(
            program, empty_startup_program, pass_name, attrs, attr_types
        )


def _fetch_var(name, scope=None, return_numpy=True):
    """
    Fetch the value of the variable with the given name from the
    given scope.

    Args:
        name(str): name of the variable. Typically, only persistable variables
            can be found in the scope used for running the program.
        scope(core.Scope|None): scope object. It should be the scope where
            you pass to Executor.run() when running your program.
            If None, global_scope() will be used. Default None.
        return_numpy(bool): whether convert the tensor to numpy.ndarray.
            Default True.

    Returns:
       LodTensor|numpy.ndarray
    """
    assert isinstance(name, str)
    if scope is None:
        scope = global_scope()
    assert isinstance(scope, core._Scope)

    var = scope.find_var(_to_name_str(name))
    assert var is not None, (
        "Cannot find " + name + " in scope. Perhaps you need to make the"
        " variable persistable by using var.persistable = True in your"
        " program."
    )
    tensor = var.get_tensor()
    if return_numpy:
        tensor = as_numpy(tensor, copy=True)
    return tensor


def _to_name_str(var):
    def _to_str(var):
        if isinstance(var, Variable):
            return var.desc.name()
        elif isinstance(var, str):
            return var
        elif isinstance(var, str):
            return str(var)
        elif isinstance(var, Operator):
            return str(id(var))
        else:
            raise TypeError(str(var) + " should be Variable, Operator or str")

    # NOTEz(zhiqiu): The item in fetch_list may be tuple returned by Optimizer.minimize(),
    # see comments in _split_optimize_ops_in_fetch_list for more details.
    if isinstance(var, tuple):
        var = var[0]
    if isinstance(var, list):
        s = [_to_str(item) for item in var]
        return ','.join(s)
    else:
        return _to_str(var)


def _prepare_fleet_executor():
    from ..distributed.backup_env import getenv_or_backup
    from ..distributed.fleet.proto import fleet_executor_desc_pb2

    trainer_endpoints_str = getenv_or_backup("PADDLE_TRAINER_ENDPOINTS", "")
    trainer_endpoints = trainer_endpoints_str.split(',')
    fleet_exe_desc = fleet_executor_desc_pb2.FleetExecutorDesc()
    cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
    fleet_exe_desc.cur_rank = cur_rank
    nrank = len(trainer_endpoints)
    for rank, endpoint in enumerate(trainer_endpoints):
        rank_info = fleet_executor_desc_pb2.RankInfo()
        rank_info.rank = rank
        rank_info.ip_port = endpoint
        fleet_exe_desc.cluster_info.append(rank_info)
    fleet_exe = core.FleetExecutor(fleet_exe_desc.SerializeToString())
    return fleet_exe


def _get_strong_program_cache_key_for_new_exe(program, scope, feed, fetch_list):
    return (
        program.desc.cached_hash_str()
        + str(scope.raw_address())
        + _get_program_cache_key(feed, fetch_list)
    )


def _get_strong_program_cache_key(program, feed, fetch_list):
    # TODO(zhiqiu): use hash_str to generate cache key as above
    def _get_varname_from_block(block):
        block_str = []
        for var_name in list(block.vars.keys()):
            block_str.append(var_name)
        return "\n".join(block_str)

    inner_program = (
        program._program
        if isinstance(program, compiler.CompiledProgram)
        else program
    )
    return (
        _get_varname_from_block(inner_program.blocks[0])
        + str(id(program))
        + _get_program_cache_key(feed, fetch_list)
    )


def _get_program_cache_key(feed, fetch_list):
    feed_var_names = []
    if isinstance(feed, dict):
        feed_var_names = list(feed.keys())
    elif isinstance(feed, (list, tuple)):
        for i, each in enumerate(feed):
            feed_var_names += list(each.keys())
    fetch_var_names = list(map(_to_name_str, fetch_list))
    return str(feed_var_names + fetch_var_names)


def _as_lodtensor(data, place, dtype=None):
    """
    Convert numpy.ndarray to Tensor, its only support Tensor without LoD information.
    For higher dimensional sequence data, please use LoDTensor directly.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle.base as base
            >>> place = base.CPUPlace()
            >>> exe = base.Executor(place)
            >>> data = np.array((100, 200, 300))
            >>> np_outs = map(lambda x: base.executor._as_lodtensor(x, place), data)

    Args:
        data(numpy.ndarray|list|tuple|scalar): a instance of array, scalar, list or tuple
        data(core.Place): the place of created tensor
        dtype(core.VarDesc.VarType|str): the expected data type of created tensor

    Returns:
        LoDTensor
    """
    # NOTE(zhiqiu): convert python builtin, like float, int, and list, to numpy ndarray
    if not isinstance(data, np.ndarray):
        assert (
            dtype is not None
        ), 'The dtype should be given when feed data is not np.ndarray'
        dtype = (
            convert_dtype(dtype)
            if isinstance(dtype, core.VarDesc.VarType)
            else dtype
        )
        if np.isscalar(data):
            data = np.array(data).astype(dtype)
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
            if data.dtype == np.object_:
                raise TypeError(
                    "\n\tFaild to convert input data to a regular ndarray :\n\t* Usually "
                    "this means the input data contains nested lists with different lengths. "
                    "Please consider using 'base.create_lod_tensor' to convert it to a LoD-Tensor."
                )
            data = data.astype(dtype)
        else:
            raise TypeError(
                f"Convert data of type {type(data)} to Tensor is not supported"
            )

    # convert numpy.ndarray to tensor
    tensor = core.LoDTensor()
    tensor.set(data, place)
    return tensor


def _can_use_interpreter_core(program, place):
    compiled = isinstance(program, compiler.CompiledProgram) or isinstance(
        program._graph, compiler.CompiledProgram
    )
    if compiled:
        compiled_program = (
            program
            if isinstance(program, compiler.CompiledProgram)
            else program._graph
        )

        # Unsupported case 1: inference
        if compiled_program._is_inference:
            warnings.warn(
                "Standalone executor is not used for inference",
                UserWarning,
            )
            return False

    return True


@lru_cache()
def _warning_once(msg):
    logging.warning(msg)


class FetchHandler:
    def __init__(self, var_dict=None, period_secs=60):
        assert var_dict is not None
        self.var_dict = var_dict
        self.period_secs = period_secs

    def handler(self, res_dict):
        for key in res_dict:
            if type(res_dict[key]) is np.ndarray:
                sys.stdout.write(f"{key}[0]: {res_dict[key][0]} ")
        sys.stdout.write("\n")

    @staticmethod
    def help():
        print(
            """
class FetchHandlerExample(FetchHandler):
    def handler(self, res_dict):
        print(res_dict["auc"])
        print("auc: {}, {}".format(res_dict["auc"], time.ctime()))

auc = Variable()
var_dict = {"auc": auc}
handler = FetchHandlerExample(var_dict=var_dict)
"""
        )


class _StandaloneExecutor:
    def __init__(self, place, plan, scope):
        self._place = core.Place()
        self._place.set_place(place)
        self._plan = plan
        self._scope = scope
        self._new_exe = self._create_new_executor()

    def run(self, feed_names, return_numpy=True):
        """
        Args:
            feed_names(list): This parameter represents the input names of the model.
            fetch_list(list): This parameter represents the Tensors that need to be returned
                after the model runs. The default is None.
            return_numpy(bool): This parameter indicates whether convert the fetched Tensors
                (the Tensor specified in the fetch list) to numpy.ndarray. if it is False,
                the type of the return value is a list of :code:`LoDTensor`. The default is True.
        """
        tensors = self._new_exe.run(feed_names)._move_to_list()
        if return_numpy:
            tensors = as_numpy(tensors, copy=True)
            return _merge_tensors(tensors, self._plan.micro_batch_num())
        else:
            if self._plan.micro_batch_num() > 1:
                raise RuntimeError(
                    "`merge_tensor` does not support when return_numpy is False."
                )
            return tensors

    def _create_new_executor(self):
        new_exe = core.StandaloneExecutor(self._place, self._plan, self._scope)

        return new_exe


class _ExecutorCache:
    class _CachedData:
        def __init__(
            self,
            program,
            feed,
            fetch_list,
            feed_var_name,
            fetch_var_name,
            place,
            scope,
        ):
            self.program = program
            self.feed = feed
            self.fetch_list = fetch_list
            self.feed_var_name = feed_var_name
            self.fetch_var_name = fetch_var_name
            self.place = place
            self.scope = scope

            # NOTE(Ruibiao): Not all changeable item is considered for key at present,
            # ONLY: program, feed, and fetch_list
            if isinstance(self.program, compiler.CompiledProgram):
                if not self.program._program:
                    # The program holds no _program, maybe it is constructed by graph.
                    # Convert graph to program in order to generate key.
                    self.program._program = framework.IrGraph(
                        self.program._graph
                    ).to_program()
                self.key = hash(
                    _get_strong_program_cache_key_for_new_exe(
                        self.program._program,
                        self.scope,
                        self.feed,
                        self.fetch_list,
                    )
                )
            else:
                self.key = hash(
                    _get_strong_program_cache_key_for_new_exe(
                        self.program, self.scope, self.feed, self.fetch_list
                    )
                )

        def __eq__(self, other):
            return (
                isinstance(other, _ExecutorCache._CachedData)
                and self.key == other.key
            )

        def __hash__(self):
            return self.key

    def __init__(self):
        # NOTE(Ruibiao): Wrap the lru_cache in constructor so that the cache is local to
        # the _ExecutorCache instance, otherwise a global cache may not be released after
        # the Executor instance deleted
        self._get_cached_program_and_executor = lru_cache(maxsize=8)(
            self._get_program_and_executor
        )

    def clear(self):
        self._get_cached_program_and_executor.cache_clear()

    def get_program_and_executor(
        self,
        program,
        feed,
        fetch_list,
        feed_var_name,
        fetch_var_name,
        place,
        scope,
    ):
        return self._get_cached_program_and_executor(
            self._CachedData(
                program,
                feed,
                fetch_list,
                feed_var_name,
                fetch_var_name,
                place,
                scope,
            )
        )

    def _get_program_and_executor(self, cached_data):
        program = cached_data.program
        inner_program = (
            program._program
            if isinstance(program, compiler.CompiledProgram)
            else program
        )
        feed = cached_data.feed
        fetch_list = cached_data.fetch_list
        feed_var_name = cached_data.feed_var_name
        fetch_var_name = cached_data.fetch_var_name
        place = cached_data.place
        scope = cached_data.scope

        # To apply IR pass, compile the Program to IrGraph and convert it back to Program
        if isinstance(program, compiler.CompiledProgram) or isinstance(
            program._graph, compiler.CompiledProgram
        ):
            compiled_program = (
                program
                if isinstance(program, compiler.CompiledProgram)
                else program._graph
            )
            build_strategy = compiled_program._build_strategy
            # print(f"Program before convert:\n {inner_program}", flush=True)
            use_cuda_graph = False
            # When using cuda graph, the cuda graph preparation logic in PE is not
            # executed, but it is processed in the constructor of new executor.
            if (
                build_strategy is not None
                and build_strategy.allow_cuda_graph_capture
            ):
                use_cuda_graph = True
                build_strategy.allow_cuda_graph_capture = False
                set_flags({"FLAGS_new_executor_use_cuda_graph": True})
            compiled_program._compile(scope, place)
            if use_cuda_graph:
                build_strategy.allow_cuda_graph_capture = True
            ir_graph = framework.IrGraph(compiled_program._graph)
            converted_program = ir_graph.to_program()

            if hasattr(inner_program, 'lr_scheduler'):
                converted_program.lr_scheduler = inner_program.lr_scheduler

            inner_program = converted_program
            # print(f"Program after convert:\n {inner_program}", flush=True)
        else:
            build_strategy = None
            from paddle.incubate.autograd import prim2orig, prim_enabled

            if prim_enabled() and program == default_main_program():
                prim2orig()

            inner_program = program

        program = _add_feed_fetch_ops(
            program=inner_program,
            feed=feed,
            fetch_list=fetch_list,
            feed_var_name=feed_var_name,
            fetch_var_name=fetch_var_name,
            use_fetch_v2=True,
        )

        # standalone executor will apply buffer_shared_inplace_pass and
        # inplace_addto_op_pass to program according to build_strategy
        enable_inplace = (
            True
            if build_strategy is None or build_strategy.enable_inplace
            else False
        )

        enable_addto = (
            True
            if build_strategy is not None and build_strategy.enable_addto
            else False
        )

        if get_flags('FLAGS_enable_new_ir_in_executor')[
            'FLAGS_enable_new_ir_in_executor'
        ]:
            # todo(phlrain), skip inplace add addto pass in new IR
            enable_inplace = False
            enable_addto = False

        if enable_inplace or enable_addto:
            # inplace should skip feed and fetch var
            skip_var_names = eval(_get_program_cache_key(feed, fetch_list))
            _apply_inplace_addto_pass(
                program, enable_inplace, enable_addto, skip_var_names
            )

        new_program = program.clone()
        if (
            new_program._pipeline_opt
            and "standalone_opt" in new_program._pipeline_opt
        ):
            from paddle.distributed.passes.pipeline_scheduler_pass import (
                apply_pass,
            )

            standalone_opt = new_program._pipeline_opt["standalone_opt"]
            pass_name = standalone_opt["schedule_mode"]
            plan = apply_pass(
                new_program, new_program, pass_name, standalone_opt
            )
        else:
            default_job = core.Job("default")
            if get_flags("FLAGS_enable_new_ir_in_executor")[
                'FLAGS_enable_new_ir_in_executor'
            ]:
                type_to_program = {
                    "default": translate_to_new_ir(new_program.desc)
                }
            else:
                type_to_program = {"default": new_program.desc}
            plan = core.Plan([default_job], type_to_program)

        new_exe = _StandaloneExecutor(place, plan, scope)
        return new_program, new_exe

    def get_pir_program_and_executor(
        self,
        program,
        feed,
        fetch_list,
        feed_var_name,
        fetch_var_name,
        place,
        scope,
    ):
        _add_pir_fetch_ops(
            program, fetch_list=fetch_list, fetch_var_name=fetch_var_name
        )

        default_job = core.Job("default")
        type_to_program = {"default": program}
        plan = core.Plan([default_job], type_to_program)

        new_exe = _StandaloneExecutor(place, plan, scope)
        return program, new_exe


class Executor:
    """
    :api_attr: Static Graph

    An Executor in Python, supports single/multiple-GPU running,
    and single/multiple-CPU running.

    Args:
        place(paddle.CPUPlace()|paddle.CUDAPlace(n)|str|None): This parameter represents
            which device the executor runs on. When this parameter is None, PaddlePaddle
            will set the default device according to its installation version. If Paddle
            is CPU version, the default device would be set to `CPUPlace()` . If Paddle is
            GPU version, the default device would be set to `CUDAPlace(0)` . Default is None.
            If ``place`` is string, it can be ``cpu``, and ``gpu:x``, where ``x``
            is the index of the GPUs. Note: users only pass one Place or None to initialize
            Executor when using multiple-cards. Other APIs will override the cards. See
            `document for multiple-cards <https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/01_paddle2.0_introduction/update_en.html#stand-alone-multi-card-launch>`_

    Returns:
        Executor

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import numpy
            >>> import os

            >>> # Executor is only used in static graph mode
            >>> paddle.enable_static()

            >>> # Set place explicitly.
            >>> # use_cuda = True
            >>> # place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            >>> # exe = paddle.static.Executor(place)

            >>> # If you don't set place, PaddlePaddle sets the default device.
            >>> exe = paddle.static.Executor()

            >>> train_program = paddle.static.Program()
            >>> startup_program = paddle.static.Program()
            >>> with paddle.static.program_guard(train_program, startup_program):
            ...     data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
            ...     hidden = paddle.static.nn.fc(data, 10)
            ...     loss = paddle.mean(hidden)
            ...     paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
            ...
            >>> # Run the startup program once and only once.
            >>> # Not need to optimize/compile the startup program.
            >>> exe.run(startup_program)

            >>> # Run the main program directly without compile.
            >>> x = numpy.random.random(size=(10, 1)).astype('float32')
            >>> loss_data, = exe.run(train_program, feed={"X": x}, fetch_list=[loss.name])

            >>> # Or, compiled the program and run. See `CompiledProgram`
            >>> # for more details.
            >>> compiled_prog = paddle.static.CompiledProgram(
            ...     train_program)
            >>> loss_data, = exe.run(compiled_prog, feed={"X": x}, fetch_list=[loss.name])

    """

    def __init__(self, place=None):
        if place is None:
            expected_place = framework._current_expected_place_()
            self.place = expected_place
        else:
            self.place = framework._get_paddle_place(place)
        self.program_caches = {}
        self.ctx_caches = {}
        self.trainer_caches = {}
        self.scope_caches = {}
        self.micro_scope_cache = {}
        self.var_caches = {}
        self.pruned_program_caches = {}
        p = core.Place()
        p.set_place(self.place)
        self._default_executor = core.Executor(p)
        self._closed = False
        self.pruned_program_scope_caches = {}
        self._prepare_to_run_called = False

        self._auto_checkpoint_name = unique_name.generate(
            "__auto_checkpoint_executor__"
        )

        self._executor_cache = _ExecutorCache()

        self._fleet_executor = None
        # TODO(liyurui): This option will be removed and always true when the functionality
        # of fleet executor with standalone executor is ready.
        self._fleet_executor_with_standalone = False

        self.op_role_key = core.op_proto_and_checker_maker.kOpRoleAttrName()

    def _is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and int(
            op.all_attrs()[self.op_role_key]
        ) & int(core.op_proto_and_checker_maker.OpRole.Optimize)

    def __del__(self):
        # NOTE(Ruibiao): The manually call of clear is required. Because in Python, executor_cache
        # may not immediately destructed after Executor instance deleted (so does not the _StandaloneExecutor),
        # that brings errors to mkl-dnn unit tests (see ClearMKLDNNCache in interpretercore.cc for why).
        self._executor_cache.clear()

    def _get_scope_cache(self, program_cache_key):
        return self.scope_caches.get(program_cache_key, None)

    def _get_ctx_cache(self, program_cache_key):
        return self.ctx_caches.get(program_cache_key, None)

    def _get_trainer_cache(self, program_cache_key):
        return self.trainer_caches.get(program_cache_key, None)

    def _get_program_cache(self, program_cache_key):
        return self.program_caches.get(program_cache_key, None)

    def _add_program_cache(self, program_cache_key, program):
        self.program_caches[program_cache_key] = program

    def _get_pruned_program_cache(self, program_cache_key):
        return self.pruned_program_caches.get(program_cache_key, None)

    def _add_pruned_program_cache(self, program_cache_key, program):
        self.pruned_program_caches[program_cache_key] = program

    def _get_pruned_program_scope_cache(self, program_cache_key):
        return self.pruned_program_scope_caches.get(program_cache_key, None)

    def _add_pruned_program_scope_cache(self, program_cache_key, program):
        self.pruned_program_scope_caches[program_cache_key] = program

    def _add_ctx_cache(self, ctx_cache_key, ctx):
        self.ctx_caches[ctx_cache_key] = ctx

    def _add_trainer_cache(self, trainer_cache_key, ctx):
        self.trainer_caches[trainer_cache_key] = ctx

    def _add_scope_cache(self, scope_cache_key, scope):
        self.scope_caches[scope_cache_key] = scope

    def _add_micro_scopes_cache(self, program_cache_key, micro_scopes: list):
        self.micro_scope_cache[program_cache_key] = micro_scopes

    def _get_micro_scopes_cache(self, program_cache_key):
        return self.micro_scope_cache.get(program_cache_key, None)

    def _log_force_set_program_cache(self, use_program_cache):
        _warning_once(
            f"use_program_cache is force set to {use_program_cache} by FLAGS_FORCE_USE_PROGRAM_CACHE"
        )

    def _feed_data(self, program, feed, feed_var_name, scope):
        # feed var to framework
        global_block = program.global_block()
        for op in global_block.ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                var = global_block.var(feed_target_name)
                if var.dtype != core.VarDesc.VarType.STRINGS:
                    if not isinstance(cur_feed, core.LoDTensor):
                        cur_feed = _as_lodtensor(
                            cur_feed, self.place, var.dtype
                        )
                    check_feed_shape_type(var, cur_feed)
                idx = op.desc.attr('col')
                pir_flag_name = 'FLAGS_enable_new_ir_in_executor'
                if get_flags(pir_flag_name)[pir_flag_name]:
                    core.set_feed_variable(
                        scope, cur_feed, feed_target_name, idx
                    )
                else:
                    micro_cur_feed = [cur_feed]
                    num_micro_batch = 1
                    if (
                        program._pipeline_opt
                        and "standalone_opt" in program._pipeline_opt
                    ):
                        num_micro_batch = program._pipeline_opt[
                            "standalone_opt"
                        ]["num_micro_batches"]
                        batch_size = (
                            cur_feed.shape()[0]
                            if callable(cur_feed.shape)
                            else cur_feed.shape[0]
                        )
                        assert batch_size % num_micro_batch == 0
                        micro_cur_feed = np.split(
                            np.array(cur_feed), num_micro_batch, 0
                        )
                    for i in range(num_micro_batch):
                        micro_feed = (
                            _as_lodtensor(
                                micro_cur_feed[i], self.place, var.dtype
                            )
                            if num_micro_batch > 1
                            else micro_cur_feed[i]
                        )
                        core.set_feed_variable(
                            scope,
                            micro_feed,
                            feed_var_name,
                            idx * num_micro_batch + i,
                        )
            else:
                break

    def _pir_feed_data(self, program, feed, scope):
        # feed var to framework
        global_block = program.global_block()
        for op in global_block.ops:
            if op.name() == 'pd_op.data':
                feed_target_name = op.attrs()["name"]
                var_type = paddle_type_to_proto_type[op.attrs()["dtype"]]
                var_shape = op.attrs()["shape"]
                cur_feed = feed[feed_target_name]
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = _as_lodtensor(cur_feed, self.place, var_type)
                pir_check_feed_shape_type(
                    cur_feed, feed_target_name, var_shape, var_type
                )
                # the last arg of set_feed_variable has no effect in pir, we pass 0 by default.
                core.set_feed_variable(scope, cur_feed, feed_target_name, 0)
            else:
                break

    def _fetch_data(self, fetch_list, fetch_var_name, scope):
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in range(len(fetch_list))
        ]
        return outs

    @classmethod
    def _split_optimize_ops_in_fetch_list(cls, fetch_list):
        """
        Split optimize_ops from fetch_list, which provided to specify program prunning.
        Args:
            fetch_list(list): The original fetch_list.
            Possible types of fetch_list are:
                fetch_list = ['loss']
                fetch_list = [[sgd, sgd], 'loss']
                fetch_list = [([sgd, sgd], [(param, grad)]), 'loss']

        Returns:
            optimize_ops(list): The optimize operators splited from fetch_list.
            fetch_list(list):  The updated fetch_list which does not contain optimize operators.
        """
        _optimize_ops = []
        _fetch_list = []

        def _get_targets(_optimize_ops, _fetch_list, item):
            if isinstance(item, Operator):
                if item._is_optimize_op():
                    _optimize_ops.append(item)
                else:
                    raise TypeError(
                        "The operator in fetch_list is not an optimize_op"
                    )
            elif isinstance(item, (Variable, str)):
                _fetch_list.append(item)
            else:
                raise TypeError(
                    "The item in fetch_list should be str, variable or optimize_op, but received %s.",
                    type(item),
                )

        for index, item in enumerate(fetch_list):
            # NOTE(zhiqiu): to support (optimizer_ops, param_and_grads) and optimizer_ops in fetch_list
            # we should handle tuple and list in fetch_list.
            # TODO(zhiqiu): find a better way to handle that.
            if isinstance(item, list):
                for i in item:
                    _get_targets(_optimize_ops, _fetch_list, i)
            elif isinstance(item, tuple):
                if not isinstance(item[0], (list, tuple)):
                    raise TypeError(
                        "Requires fetch_list[{}][0] shall be one of (list, tuple) when type(fetch_list[{}]) is `tuple`, but received fetch_list[{}][0]'s type is `{}`.".format(
                            index, index, index, type(item[0]).__name__
                        )
                    )
                for i in item[0]:
                    _get_targets(_optimize_ops, _fetch_list, i)
            else:
                _get_targets(_optimize_ops, _fetch_list, item)

        return _fetch_list, _optimize_ops

    @classmethod
    def _prune_program(
        cls, program, feed=None, fetch_list=None, optimize_ops=None
    ):
        """
        Prune operators and variables which are not needed to generate
        :code:`fetch_list` and optimize operators.
        Prune operators and variables which are needed
        to generate variables to be feeded.

        Notes: This is a very low level API. Users should not use this API
        directly.

        Args:
            program(Program): the origin program
            feed(list|dict): feed dict or list.
            fetch_list(list|Variable): A list of variables need to be fetched
            optimize_ops(list[Operator]): A list of optimizer operators

        Returns:
            Program:  A new, pruned program.
        """
        compiled = isinstance(program, compiler.CompiledProgram)
        if compiled:
            if program._program:
                origin_program = program._program
            else:
                warnings.warn(
                    "The program holds no _program, maybe it is constructed by graph, which can't be pruned yet."
                )
                return
        else:
            origin_program = program

        feed_names = []
        if isinstance(feed, dict):
            feed_names = list(feed.keys())
        elif isinstance(feed, (list, tuple)):
            for i, each in enumerate(feed):
                feed_names += list(each.keys())

        # if optimize_ops is [], all optimize ops in the program is used.
        if not optimize_ops:
            for block in origin_program.blocks:
                for op in block.ops:
                    if op._is_optimize_op():
                        optimize_ops.append(op)

        targets = fetch_list + optimize_ops
        pruned_program = origin_program._prune_with_input(feed_names, targets)

        if compiled:
            # for compiled program, update the underlying program, re-generate graph,
            # and reset the flag so it can be compiled again.
            program._program = pruned_program
            program._graph = core.Graph(pruned_program.desc)
            program._compiled = False
        else:
            program = pruned_program

        return program

    @classmethod
    def _update_feed(cls, program, feed):
        """
        Update the feed dict, remove the feed item which is pruned in program.

        Notes: This is a very low level API. Users should not use this API
        directly.

        Args:
            program(Program): the pruned program.
            feed(list|dict): feed dict or list.

        Returns:
            feed:(list|dict)  updated feed.
        """
        compiled = isinstance(program, compiler.CompiledProgram)
        if compiled:
            if program._program:
                global_block = program._program.global_block()
            else:
                warnings.warn(
                    "The program holds no _program, maybe it is constructed by graph."
                )
                return feed
        else:
            global_block = program.global_block()

        if isinstance(feed, dict):
            for feed_name in list(feed.keys()):
                if not global_block.has_var(feed_name):
                    feed.pop(feed_name)
                    warnings.warn(
                        "The variable %s is not found in program. It is not declared or is pruned."
                        % feed_name
                    )

        elif isinstance(feed, (list, tuple)):
            for i, each in enumerate(feed):
                for feed_name in list(each.keys()):
                    if not global_block.has_var(feed_name):
                        each.pop(feed_name)
                        warnings.warn(
                            "The variable %s is not found in program. It is not declared or is pruned."
                            % feed_name
                        )
        return feed

    '''
    TODO(typhoonzero): Define "no longer use" meaning? Can user create
    a new Executor for the same program and run?
    TODO(panyx0718): Why ParallelExecutor doesn't have close?
    '''

    def close(self):
        """
        Close the executor. This interface is used for distributed training (PServers mode).
        This executor can not be used after calling the interface, because
        this interface releases resources associated with the current Trainer.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle

                >>> cpu = paddle.CPUPlace()
                >>> exe = paddle.static.Executor(cpu)
                >>> # execute training or testing
                >>> exe.close()
        """
        if not self._closed:
            self._closed = True
            for k, trainer_instance in self.trainer_caches.items():
                self._default_executor.release_trainer(trainer_instance)
                del trainer_instance
            self._default_executor.close()

    def run(
        self,
        program=None,
        feed=None,
        fetch_list=None,
        feed_var_name='feed',
        fetch_var_name='fetch',
        scope=None,
        return_numpy=True,
        use_program_cache=False,
        use_prune=False,
    ):
        """
        Run the specified :code:`Program` or :code:`CompiledProgram`. It should be noted that the executor
        will execute all the operators in :code:`Program` or :code:`CompiledProgram` without pruning some
        operators of the :code:`Program` or :code:`CompiledProgram` according to fetch_list. And you could
        specify the scope to store the :code:`Tensor` during the executor running if the scope
        is not set, the executor will use the global scope, i.e. :code:`paddle.static.global_scope()`.

        Args:
            program(Program|CompiledProgram): This parameter represents the :code:`Program` or
                :code:`CompiledProgram` to be executed. If this parameter is not provided, that
                parameter is None, the program will be set to :code:`paddle.static.default_main_program()`.
                The default is None.
            feed(list|dict): This parameter represents the input Tensors of the model.
                If it is single card training, the feed is dict type, and if it is multi-card
                training, the parameter feed can be dict or list of Tensors. If the
                parameter type is dict, the data in the feed will be split and sent to
                multiple devices (CPU/GPU), that is to say, the input data will be evenly
                sent to different devices, so you should make sure the number of samples of
                the current mini-batch must be greater than the number of places;
                if the parameter type is list, those data are copied directly to each device,
                so the length of this list should be equal to the number of places.
                The default is None.
            fetch_list(list): This parameter represents the Tensors that need to be returned
                after the model runs. The default is None.
            feed_var_name(str): This parameter represents the name of the input Tensor of
                the feed operator. The default is "feed".
            fetch_var_name(str): This parameter represents the name of the output Tensor of
                the fetch operator. The default is "fetch".
            scope(Scope): the scope used to run this program, you can switch
                it to different scope. default is :code:`paddle.static.global_scope()`
            return_numpy(bool): This parameter indicates whether convert the fetched Tensors
                (the Tensor specified in the fetch list) to numpy.ndarray. if it is False,
                the type of the return value is a list of :code:`LoDTensor`. The default is True.
            use_program_cache(bool): This parameter indicates whether the input :code:`Program` is cached.
                If the parameter is True, the model may run faster in the following cases:
                the input program is :code:`paddle.static.Program`, and the parameters(program, feed Tensor name
                and fetch_list Tensor) of this interface remains unchanged during running.
                The default is False.
            use_prune(bool): This parameter indicates whether the input :code:`Program` will be pruned.
                If the parameter is True, the program will be pruned accroding to the given feed and fetch_list,
                which means the operators and variables in program that generate :code:`feed` and are not
                needed to generate :code:`fetch_list` will be pruned. The default is False, which means the
                program will not pruned and all the operators and variables will be executed during running.
                Note that if the tuple returned from :code:`Optimizer.minimize()` is passed to :code:`fetch_list`,
                :code:`use_prune` will be overrided to True, and the program will be pruned.

        Returns:

            List: The fetched result list.

        Examples:

            .. code-block:: python
                :name: code-example-1

                >>> import paddle
                >>> import numpy

                >>> # First create the Executor.
                >>> paddle.enable_static()
                >>> place = paddle.CPUPlace()  # paddle.CUDAPlace(0)
                >>> exe = paddle.static.Executor(place)

                >>> data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                >>> hidden = paddle.static.nn.fc(data, 10)
                >>> loss = paddle.mean(hidden)
                >>> adam = paddle.optimizer.Adam()
                >>> adam.minimize(loss)
                >>> i = paddle.zeros(shape=[1], dtype='int64')
                >>> array = paddle.tensor.array_write(x=loss, i=i)

                >>> # Run the startup program once and only once.
                >>> exe.run(paddle.static.default_startup_program())

                >>> x = numpy.random.random(size=(10, 1)).astype('float32')
                >>> loss_val, array_val = exe.run(feed={'X': x},
                ...                                 fetch_list=[loss.name, array.name])
                >>> print(array_val)
                >>> # doctest: +SKIP("Random output")
                [array(0.16870381, dtype=float32)]
                >>> # doctest: -SKIP

            .. code-block:: python
                :name: code-example-2

                >>> # doctest: +REQUIRES(env:GPU)
                >>> import paddle
                >>> import numpy as np

                >>> # First create the Executor.
                >>> paddle.enable_static()
                >>> place = paddle.CUDAPlace(0)
                >>> exe = paddle.static.Executor(place)

                >>> data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                >>> class_dim = 2
                >>> prediction = paddle.static.nn.fc(data, class_dim)
                >>> loss = paddle.mean(prediction)
                >>> adam = paddle.optimizer.Adam()
                >>> adam.minimize(loss)

                >>> # Run the startup program once and only once.
                >>> exe.run(paddle.static.default_startup_program())
                >>> build_strategy = paddle.static.BuildStrategy()
                >>> binary = paddle.static.CompiledProgram(
                ...     paddle.static.default_main_program(), build_strategy=build_strategy)
                >>> batch_size = 6
                >>> x = np.random.random(size=(batch_size, 1)).astype('float32')

                >>> prediction, = exe.run(binary,
                ...                         feed={'X': x},
                ...                     fetch_list=[prediction.name])
                >>> # If the user uses two GPU cards to run this python code, the printed result will be
                >>> # (6, class_dim). The first dimension value of the printed result is the batch_size.
                >>> print("The prediction shape: {}".format(
                ...     np.array(prediction).shape))
                The prediction shape: (6, 2)

                >>> print(prediction)
                >>> # doctest: +SKIP("Random output")
                [[-0.37789783 -0.19921964]
                 [-0.3577645  -0.18863106]
                 [-0.24274671 -0.12814042]
                 [-0.24635398 -0.13003758]
                 [-0.49232286 -0.25939852]
                 [-0.44514108 -0.2345845 ]]
                >>> # doctest: -SKIP

        """
        # Temporary FLAGS, just for testing the performance of program cache
        force_use_program_cache = os.environ.get(
            'FLAGS_FORCE_USE_PROGRAM_CACHE', None
        )
        if force_use_program_cache is not None:
            use_program_cache = force_use_program_cache in [
                1,
                '1',
                True,
                'True',
                'true',
            ]
            self._log_force_set_program_cache(use_program_cache)
        if in_pir_mode():
            res = self._run_pir_impl(
                program=program,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name,
                scope=scope,
                return_numpy=return_numpy,
            )
        else:
            res = self._run_impl(
                program=program,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name,
                scope=scope,
                return_numpy=return_numpy,
                use_program_cache=use_program_cache,
                use_prune=use_prune,
            )
            core.update_autotune_status()
        return res

    def _run_impl(
        self,
        program,
        feed,
        fetch_list,
        feed_var_name,
        fetch_var_name,
        scope,
        return_numpy,
        use_program_cache,
        use_prune,
    ):
        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        use_default_main_program = program is None
        if program is None:
            program = default_main_program()

        fetch_list = self._check_fetch_list(fetch_list)

        from paddle.distributed.auto_parallel.static.utils import (
            use_new_executor,
        )

        if (
            isinstance(program, Program)
            and program._pipeline_opt
            and not use_new_executor()
        ):
            if "fleet_opt" in program._pipeline_opt:
                # Move prepare here for port conflict with nccl in startup program
                if self._fleet_executor is None:
                    self._fleet_executor = _prepare_fleet_executor()
                return self._run_using_fleet_executor(
                    program=program,
                    feed=feed,
                    fetch_list=fetch_list,
                    with_standalone_executor=self._fleet_executor_with_standalone,
                    return_numpy=return_numpy,
                )
            if "startup_program" in program._pipeline_opt:
                program = program._pipeline_opt["startup_program"]
            else:
                return self._run_pipeline(
                    program,
                    fetch_list=fetch_list,
                    use_program_cache=use_program_cache,
                )

        if isinstance(program, Program) and program._heter_pipeline_opt:
            # print("program._heter_pipeline_opt: {}".format(
            #    program._heter_pipeline_opt))
            ## change default executor
            heter_place = program._heter_pipeline_opt["heter_place"]
            heter_place = framework._get_paddle_place(heter_place)
            p = core.Place()
            p.set_place(heter_place)
            self._default_executor = core.Executor(p)
            # TODO(zhangminxu): support heterps pipeline training using exe.run
            if "startup_program" in program._heter_pipeline_opt:
                # print("get startup_program from _pipeline_opt")
                program = program._heter_pipeline_opt["startup_program"]

        if (
            isinstance(program, Program)
            and len(program.global_block().ops) == 0
        ):
            if use_default_main_program:
                error_info = (
                    "Now you are using default_main_program, "
                    "but there are no operators in the program to be executed. "
                    "Please ensure you create model correctly or you can pass "
                    "the Program or the CompiledProgram manually."
                )
                warnings.warn(error_info)

        if scope is None:
            scope = global_scope()

        # use_prune can be overrided by putting optimize_ops in fetch_list
        _origin_fetch_list = fetch_list
        _origin_program = program
        fetch_list, optimize_ops = self._split_optimize_ops_in_fetch_list(
            fetch_list
        )
        if optimize_ops:
            use_prune = True
        if use_prune:
            cache_key = _get_strong_program_cache_key(
                program, feed, _origin_fetch_list
            )
            cached_pruned_program = self._get_pruned_program_cache(cache_key)
            if cached_pruned_program is None:
                if isinstance(program, compiler.CompiledProgram):
                    program_scope_cache = self._get_pruned_program_scope_cache(
                        str(id(_origin_program))
                    )
                    # copy the original program, so it can be cached.
                    program = copy.copy(program)
                    # share the local scopes for same original CompiledProgram.
                    program._share_vars_from = program_scope_cache
                    if (
                        self._get_pruned_program_scope_cache(
                            str(id(_origin_program))
                        )
                        is None
                    ):
                        self._add_pruned_program_scope_cache(
                            str(id(_origin_program)), program
                        )
                pruned_program = self._prune_program(
                    program, feed, fetch_list, optimize_ops
                )
                self._add_pruned_program_cache(cache_key, pruned_program)
            else:
                pruned_program = cached_pruned_program

            feed = self._update_feed(pruned_program, feed)
            program = pruned_program

        if _can_use_interpreter_core(program, self.place):
            if feed is None:
                feed = {}
            elif isinstance(feed, (list, tuple)):
                assert len(feed) == 1, "Not compiled with data parallel"
                feed = feed[0]
            if not isinstance(feed, dict):
                raise TypeError(
                    "feed requires dict as its Parameter. But you passed in %s"
                    % (type(feed))
                )
            feed = self._update_feed(program, feed)

            stored_flag = {}
            if isinstance(program, compiler.CompiledProgram) or isinstance(
                program._graph, compiler.CompiledProgram
            ):
                compiled_program = (
                    program
                    if isinstance(program, compiler.CompiledProgram)
                    else program._graph
                )
                build_strategy = compiled_program._build_strategy
                if build_strategy is not None and build_strategy.sequential_run:
                    schedule_flag = [
                        'FLAGS_new_executor_serial_run',
                        'FLAGS_new_executor_sequential_run',
                    ]
                    for flag in schedule_flag:
                        value = os.getenv(flag, False)
                        if isinstance(value, str):
                            value = value.lower()
                            value = True if value == 'true' else False
                        stored_flag[flag] = bool(value)
                    set_flags({f: True for f in schedule_flag})

            program, new_exe = self._executor_cache.get_program_and_executor(
                program,
                feed,
                fetch_list,
                feed_var_name,
                fetch_var_name,
                self.place,
                scope,
            )

            self._feed_data(program, feed, feed_var_name, scope)

            if hasattr(program, 'lr_scheduler'):
                from paddle.optimizer.lr import LRScheduler

                assert isinstance(
                    program.lr_scheduler, LRScheduler
                ), "must be LRScheduler"
                lr_scheduler = program.lr_scheduler
                lr_value = lr_scheduler()
                lr_var = program.global_block().vars[lr_scheduler._var_name]
                data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
                tensor = core.get_variable_tensor(scope, lr_scheduler._var_name)
                # NOTE(dev): `tensor.set(data, self.place)` always call TensorCopySync that is a blocking behavior. So we use `_copy_from` to replace it.
                cpu_tensor = _as_lodtensor(data, core.CPUPlace())
                if core.is_cuda_graph_capturing():
                    warnings.warn(
                        "Caution!!! When capturing CUDA Graph, the learning rate scheduler would not "
                        "take any effect! Please set the learning rate manually before each batch!"
                    )
                elif core.is_compiled_with_ipu():
                    # for ipu, tensor is allocated on cpu
                    tensor._copy_from(cpu_tensor, tensor._place())
                else:
                    tensor._copy_from(cpu_tensor, self.place)

            ret = new_exe.run(list(feed.keys()), return_numpy)
            set_flags(stored_flag)
            return ret

        compiled = isinstance(program, compiler.CompiledProgram)

        # Check if paddle.static.data() variable no feed data
        if use_prune:
            if compiled:
                global_block = program._program.global_block()
            else:
                global_block = program.global_block()
            for varname in global_block.vars:
                vardesc = global_block.desc.find_var(varname.encode())
                varobj = global_block.vars[varname]

                if (
                    vardesc.persistable() == False
                    and vardesc.type() == core.VarDesc.VarType.LOD_TENSOR
                    and vardesc.need_check_feed() == True
                    and varobj.stop_gradient == True
                    and varobj.is_data == True
                    and varobj.belong_to_optimizer == False
                    and varname not in feed
                ):
                    raise ValueError('Need feed data for variable %s' % varname)

        acp._auto_checkpoint(self, program)

        program._compile(scope, self.place)
        assert (
            program._is_inference
        ), f"Program must have _is_inference = True, but get {program._is_inference}"
        return self._run_inference(program._executor, feed)

    def _run_pir_impl(
        self,
        program,
        feed,
        fetch_list,
        feed_var_name,
        fetch_var_name,
        scope,
        return_numpy,
    ):
        import paddle

        Program = paddle.pir.Program
        default_main_program = paddle.pir.core.default_main_program

        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        use_default_main_program = program is None
        if use_default_main_program:
            program = default_main_program()

        fetch_list = self._check_fetch_list(fetch_list)

        if (
            isinstance(program, Program)
            and len(program.global_block().ops) == 0
        ):
            if use_default_main_program:
                error_info = (
                    "Now you are using default_main_program, "
                    "but there are no operators in the program to be executed. "
                    "Please ensure you create model correctly or you can pass "
                    "the Program or the CompiledProgram manually."
                )
                warnings.warn(error_info)

        if scope is None:
            scope = global_scope()

        if feed is None:
            feed = {}
        elif isinstance(feed, (list, tuple)):
            assert len(feed) == 1, "Not compiled with data parallel"
            feed = feed[0]
        if not isinstance(feed, dict):
            raise TypeError(
                "feed requires dict as its Parameter. But you passed in %s"
                % (type(feed))
            )

        program, new_exe = self._executor_cache.get_pir_program_and_executor(
            program,
            feed,
            fetch_list,
            feed_var_name,
            fetch_var_name,
            self.place,
            scope,
        )

        self._pir_feed_data(program, feed, scope)

        ret = new_exe.run(list(feed.keys()), return_numpy)
        return ret

    def _run_inference(self, exe, feed):
        return exe.run(feed)

    def _check_fetch_list(self, fetch_list):
        is_fetch_var = lambda var: isinstance(var, (Variable, str, OpResult))
        is_tuple_list = lambda var: isinstance(var, (tuple, list))

        if fetch_list is None:
            return []
        if is_fetch_var(fetch_list):
            return [fetch_list]

        assert is_tuple_list(fetch_list), (
            "Currently , The fetch_list type only should be list or tuple, \n"
            "but the input type is {}. For more information please refer to \n"
            "the executor.run(...).".format(type(fetch_list))
        )

        res = []
        for i, var in enumerate(fetch_list):
            if is_fetch_var(var):
                res.append(var)
            # such as [x, 'mean_out', loss]
            elif is_tuple_list(var):
                if all(is_fetch_var(v) for v in var):
                    res.extend(list(var))
                else:
                    res.append(var)
            else:
                raise TypeError(
                    "Require fetch_list[{}] 's type shall be one of (Variable, str), but received {}.".format(
                        i, type(var).__name__
                    )
                )

        return res

    def _dump_debug_info(self, program=None, trainer=None):
        with open(str(id(program)) + "_train_desc.prototxt", "w") as fout:
            fout.write(str(trainer))
        if program._fleet_opt and "fleet_desc" in program._fleet_opt:
            with open("fleet_desc.prototxt", "w") as fout:
                fout.write(str(program._fleet_opt["fleet_desc"]))

    def _adjust_pipeline_resource(self, pipeline_opt, dataset, pipeline_num):
        filelist_length = len(dataset.dataset.get_filelist())
        if filelist_length < pipeline_num:
            pipeline_num = filelist_length
            print(
                "Pipeline training: setting the pipeline num to %d is enough because there are only %d files"
                % (filelist_length, filelist_length)
            )
        if filelist_length < pipeline_num * pipeline_opt["concurrency_list"][0]:
            print(
                "Pipeline training: setting the 1st element in concurrency_list to %d is enough because there are only %d files"
                % (filelist_length // pipeline_num, filelist_length)
            )
            pipeline_opt["concurrency_list"][0] = (
                filelist_length // pipeline_num
            )
        dataset.set_thread(pipeline_opt["concurrency_list"][0] * pipeline_num)
        return pipeline_num

    def split_program_by_device(self, program):
        ops_list = []
        type_list = []
        pre = None
        type_cpu = "cpu"
        for op in program.global_block().ops:
            if self._is_optimizer_op(op):
                break
            if op.has_attr("op_device"):
                cur_attr = (
                    op.attr("op_device")
                    if op.attr("op_device") != ""
                    else type_cpu
                )
                if pre is None or pre != cur_attr:
                    ops_list.append([])
                    type_list.append(cur_attr)
                ops_list[-1].append(op)
                pre = cur_attr
        l = len(type_list)
        i = 0
        type_heter = None
        while i < l:
            while i < l and type_list[i] == type_cpu:
                i += 1
            if i == l:
                break

            type_heter = type_list[i]
            i += 1
            start = i
            valid = True
            while i < l and type_list[i] != type_heter:
                if type_list[i] != type_cpu:
                    valid = False
                    break
                i += 1

            if i == l:
                break
            elif not valid:
                continue

            for j in range(start, i):
                for op in ops_list[j]:
                    op._set_attr("op_device", type_heter)
                type_list[j] = type_heter
                j += 1

        pre = None
        merged_ops_list = []
        merged_type_list = []
        for i in range(l):
            if pre is None or pre != type_list[i]:
                merged_ops_list.append([])
                merged_type_list.append(type_list[i])
            merged_ops_list[-1].extend(ops_list[i])
            pre = type_list[i]

        data_vars = set()
        for k in program.global_block().vars:
            var = program.global_block().var(k)
            if not var.persistable:
                data_vars.add(var.name)

        l = len(merged_ops_list)
        inputs_pre = set()
        outputs_pre = set()
        in_from_pre = [[] for i in range(l)]
        for i in range(l):
            inputs = set()
            outputs = set()
            for op in merged_ops_list[i]:
                for input in op.input_names:
                    for tmp in op.input(input):
                        if tmp not in outputs:
                            inputs.add(tmp)
                for output in op.output_names:
                    for tmp in op.output(output):
                        outputs.add(tmp)
            if i == 0:
                in_from_pre[i] = []
            elif i == 1:
                in_from_pre[i] = (outputs_pre | data_vars) & inputs
            else:
                in_from_pre[i] = outputs_pre & inputs
            inputs_pre = copy.deepcopy(inputs)
            outputs_pre = copy.deepcopy(outputs)

        l = len(in_from_pre)
        start_list = []
        end_list = []
        send_list = [[] for i in range(l)]
        sum = 0
        program_list = []
        for i in range(l):
            start_list.append(sum)
            end_list.append(sum + len(merged_ops_list[i]) - 1)
            sum += len(merged_ops_list[i])
            if i < l - 1:
                send_list[i].extend(list(in_from_pre[i + 1]))
            prog = program.clone()
            if merged_type_list[i] != type_cpu:
                prog = prog._prune_with_input(
                    list(in_from_pre[i]), list(send_list[i])
                )
                program_list.append(prog)
            else:
                program_list.append(prog)
        recv_list = [list(i) for i in in_from_pre]
        found = False
        heter_index = None
        for i in range(len(merged_type_list)):
            t = merged_type_list[i]
            if t != type_cpu:
                if found:
                    print("only one region of program can be heter")
                found = True
                heter_index = i
        if heter_index is None:
            print("warning: non heter program")
            return None
        else:
            return [
                start_list[heter_index],
                end_list[heter_index],
                send_list[heter_index],
                recv_list[heter_index],
                program_list[heter_index],
            ]

    def _prepare_trainer(
        self,
        program=None,
        dataset=None,
        scope=None,
        thread=0,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
    ):
        is_heter = 0
        use_ps_gpu = 0
        if not program._fleet_opt is None:
            if program._fleet_opt.get("worker_class", "") == "HeterCpuWorker":
                is_heter = 1
            if program._fleet_opt.get("trainer", "") == "HeterXpuTrainer":
                is_heter = 1
            if program._fleet_opt.get("use_ps_gpu", False):
                use_ps_gpu = True
        if scope is None:
            scope = global_scope()
        if fetch_list is None:
            fetch_list = []
        if fetch_info is None:
            fetch_info = []
        assert len(fetch_list) == len(fetch_info)
        compiled = isinstance(program, compiler.CompiledProgram)
        if is_heter:
            ret = self.split_program_by_device(program)
        if not compiled:
            # TODO: Need a better way to distinguish and specify different execution mode
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program._pipeline_opt
                )
            elif program._heter_pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program._heter_pipeline_opt
                )
            else:
                trainer = TrainerFactory()._create_trainer(program._fleet_opt)
                trainer._set_thread_barrier(program._is_distributed)
            trainer._set_program(program)
            if is_heter:
                trainer._set_heter_info(ret)
        else:
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program.program._pipeline_opt
                )
            elif program._heter_pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program.program._heter_pipeline_opt
                )
            else:
                trainer = TrainerFactory()._create_trainer(
                    program.program._fleet_opt
                )
            trainer._set_program(program.program)

        if thread <= 0:
            if use_ps_gpu:
                trainer._set_thread(len(program._fleet_opt["worker_places"]))
            elif dataset.thread_num <= 0:
                raise RuntimeError(
                    "You should set thread num first, either in Dataset"
                    "or in Executor.train_from_dataset"
                )
            else:
                trainer._set_thread(dataset.thread_num)
        else:
            trainer._set_thread(thread)

        trainer._set_debug(debug)
        trainer._set_fetch_var_and_info(fetch_list, fetch_info, print_period)
        return scope, trainer

    def _run_from_dataset(
        self,
        program=None,
        dataset=None,
        scope=None,
        thread=0,
        is_infer=False,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
        fetch_handler=None,
    ):
        if program._pipeline_opt is not None:
            import paddle

            if dataset is not None:
                raise RuntimeError("dataset should be None for pipeline mode")
            # The following fake dataset is created to call
            # the _prepare_trainer api, and it is meaningless.
            data_vars = []
            for var in program.global_block().vars.values():
                if var.is_data:
                    data_vars.append(var)
            dataset = paddle.base.DatasetFactory().create_dataset(
                'FileInstantDataset'
            )
            dataset.set_batch_size(1)
            dataset.set_thread(1)
            dataset.set_filelist(['None'])
            dataset.set_use_var(data_vars)
        elif program._heter_pipeline_opt is not None:
            stage_id = program._heter_pipeline_opt["pipeline_stage"]
            # print("test_fl_stage_id: {}".format(stage_id))
            heter_place = program._heter_pipeline_opt["heter_place"]
            if stage_id != 0:
                if "is_fl_mode" not in program._heter_pipeline_opt:
                    import paddle

                    if dataset is not None:
                        raise RuntimeError(
                            "dataset should be None for heter pipeline mode"
                        )
                    # The following fake dataset is created to call
                    # the _prepare_trainer api, and it is meaningless.
                    data_vars = []
                    for var in program.global_block().vars.values():
                        if var.is_data:
                            data_vars.append(var)
                    dataset = paddle.base.DatasetFactory().create_dataset(
                        'InMemoryDataset'
                    )
                    dataset.set_batch_size(1)
                    dataset.set_thread(1)
                    dataset.set_filelist(['None'])
                    dataset.set_use_var(data_vars)
            else:
                if dataset is None:
                    raise RuntimeError(
                        "dataset is need and should be initialized"
                    )
            ## change default executor
            heter_place = framework._get_paddle_place(heter_place)
            p = core.Place()
            p.set_place(heter_place)
            self._default_executor = core.Executor(p)
        else:
            if dataset is None:
                raise RuntimeError("dataset is need and should be initialized")

        dataset._prepare_to_run()
        real_fetch_list = []
        if program._pipeline_opt:
            real_program = program._pipeline_opt["section_program"]
            for fetch_var in fetch_list:
                if isinstance(fetch_var, Variable):
                    fetch_var_name = fetch_var.name
                else:
                    fetch_var_name = fetch_var
                if fetch_var_name in real_program.global_block().vars:
                    real_fetch_list.append(fetch_var)

            program._pipeline_opt["section_program"] = _add_feed_fetch_ops(
                program=program._pipeline_opt["section_program"],
                feed=[],
                fetch_list=real_fetch_list,
                feed_var_name='feed',
                fetch_var_name='fetch',
            )
            main_block = program._pipeline_opt["section_program"].block(0)
            for op in main_block.ops:
                # set the op_role of fetch op to Optimize to avoid
                # erase the fetched vars by gc for pipeline
                if op.type == 'fetch':
                    op._set_attr(
                        'op_role',
                        core.op_proto_and_checker_maker.OpRole.Optimize,
                    )
            fetch_list = None
        scope, trainer = self._prepare_trainer(
            program=program,
            dataset=dataset,
            scope=scope,
            thread=thread,
            debug=debug,
            fetch_list=fetch_list,
            fetch_info=fetch_info,
            print_period=print_period,
        )

        trainer._set_infer(is_infer)
        trainer._gen_trainer_desc()

        if program._pipeline_opt is None:
            if program._heter_pipeline_opt is None:
                self._dump_debug_info(program=program, trainer=trainer)
        # warning if dataset not set psgpu in psgpu mode
        if dataset.use_ps_gpu is False and trainer.proto_desc.use_ps_gpu:
            logging.warning("dataset should call set_use_ps_gpu in PsGpu mode")

        dataset._dynamic_adjust_before_train(trainer.proto_desc.thread_num)

        if program._heter_pipeline_opt is None:
            trainer_instance = (
                self._default_executor.init_for_dataset(  # -->InitForDataset
                    program.desc, trainer._desc(), scope, dataset.dataset
                )
            )
        else:
            # cache trainer instance for heterps pipeline training
            if fetch_list is None:
                fetch_list = []
            cache_key = _get_strong_program_cache_key(program, None, fetch_list)
            trainer_instance = self._get_trainer_cache(cache_key)
            if trainer_instance is None:
                trainer_instance = self._default_executor.init_for_dataset(
                    program.desc, trainer._desc(), scope, dataset.dataset
                )
                # print("test_fl_ps - trainer_desc: {}\n".format(trainer))
                self._add_trainer_cache(cache_key, trainer_instance)
            else:
                trainer_instance.ResetDataset(dataset.dataset)

        if fetch_handler is not None:
            scope0 = trainer_instance.get_worker_scope(0)
            fetch_monitor = FetchHandlerMonitor(scope0, fetch_handler)
            fetch_monitor.start()
            self._default_executor.run_from_dataset(trainer_instance)
            fetch_monitor.stop()
            if program._heter_pipeline_opt is None:
                self._default_executor.release_trainer(trainer_instance)
        else:
            self._default_executor.run_from_dataset(trainer_instance)
            if program._heter_pipeline_opt is None:
                self._default_executor.release_trainer(trainer_instance)

        dataset._dynamic_adjust_after_train()
        dataset._finish_to_run()
        if real_fetch_list:
            arr = scope.find_var('fetch').get_fetch_list()
            tensors = arr._move_to_list()
            return as_numpy(tensors)

        return None

    def _prepare_pipeline_ctx(
        self,
        program=None,
        dataset=None,
        scope=None,
        thread=0,
        is_infer=False,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
        fetch_handler=None,
        use_program_cache=False,
    ):
        assert program._pipeline_opt is not None
        assert dataset is None, "dataset should be None for pipeline mode"

        cache_key = _get_strong_program_cache_key(program, None, fetch_list)
        ctx = self._get_ctx_cache(cache_key)
        if use_program_cache and ctx is not None:
            return ctx

        import paddle

        # The following fake dataset is created to call
        # the _prepare_trainer api, and it is meaningless.
        def _get_dataset():
            data_vars = []
            for var in program.global_block().vars.values():
                if var.is_data:
                    data_vars.append(var)
            dataset = paddle.base.DatasetFactory().create_dataset(
                'FileInstantDataset'
            )
            dataset.set_batch_size(1)
            dataset.set_thread(1)
            dataset.set_filelist(['None'])
            dataset.set_use_var(data_vars)
            dataset._prepare_to_run()
            return dataset

        dataset = _get_dataset()

        def _get_real_program_fetch_list():
            real_program = program._pipeline_opt["section_program"]
            real_fetch_list = []
            for fetch_var in fetch_list:
                if isinstance(fetch_var, Variable):
                    fetch_var_name = fetch_var.name
                else:
                    fetch_var_name = fetch_var
                if fetch_var_name in real_program.global_block().vars:
                    real_fetch_list.append(fetch_var)

            real_program = _add_feed_fetch_ops(
                program=real_program,
                feed=[],
                fetch_list=real_fetch_list,
                feed_var_name='feed',
                fetch_var_name='fetch',
            )
            main_block = real_program.block(0)
            for op in main_block.ops:
                # set the op_role of fetch op to Optimize to avoid
                # erase the fetched vars by gc for pipeline
                if op.type == 'fetch':
                    op._set_attr(
                        'op_role',
                        core.op_proto_and_checker_maker.OpRole.Optimize,
                    )
            return real_program, real_fetch_list

        real_program, real_fetch_list = _get_real_program_fetch_list()

        program._pipeline_opt["section_program"] = real_program
        fetch_list = None

        scope, trainer = self._prepare_trainer(
            program=program,
            dataset=dataset,
            scope=scope,
            thread=thread,
            debug=debug,
            fetch_list=fetch_list,
            fetch_info=fetch_info,
            print_period=print_period,
        )

        trainer._set_infer(is_infer)
        trainer._gen_trainer_desc()

        # NOTE: only for debug, very slow
        # self._dump_debug_info(program=program, trainer=trainer)

        # warning if dataset not set psgpu in psgpu mode
        if dataset.use_ps_gpu is False and trainer.proto_desc.use_ps_gpu:
            logging.warning("dataset should call set_use_ps_gpu in PsGpu mode")
        dataset._dynamic_adjust_before_train(trainer.proto_desc.thread_num)

        trainer_desc = trainer._desc()  # slow, cache
        trainer_instance = self._default_executor.init_for_dataset(
            program.desc, trainer_desc, scope, dataset.dataset
        )

        ctx = [scope, real_fetch_list, trainer_instance]
        if use_program_cache:
            self._add_ctx_cache(cache_key, ctx)

        return ctx

    def _prepare_fleet_executor_carrier(
        self,
        carrier_id="",
        program=None,
        scope=None,
        fleet_opt=None,
        micro_scope_list=[],
        with_standalone_executor=False,
    ):
        num_micro_batches = (
            fleet_opt["num_micro_batches"]
            if "num_micro_batches" in fleet_opt
            else 1
        )
        cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "").split(',')
        nrank = len(trainer_endpoints)

        assert 'scheduler' in fleet_opt or 'tasks' in fleet_opt, (
            "Fleet executor need configuration for scheduler, you can choose from 1F1B or Origin. "
            "Or you can provide a list of task nodes to init fleet executor directly."
        )
        if 'tasks' in fleet_opt:
            assert 'task_id_to_rank' in fleet_opt, (
                "If you provide tasks to init fleet executor,"
                " task_id_to_rank should also be provided."
            )
            print('fleet executor will use user defined task nodes')
            tasks = [task.task_node() for task in fleet_opt['tasks']]
            task_id_to_rank = fleet_opt['task_id_to_rank']
        else:
            scheduler = fleet_opt['scheduler']
            if scheduler == '1F1B':
                from paddle.distributed.fleet.fleet_executor_utils import (
                    run1f1b,
                )

                if (
                    "dist_strategy" not in fleet_opt
                    or "pp_degree" not in fleet_opt["dist_strategy"]
                    or fleet_opt["dist_strategy"]["pp_degree"] == 1
                ):
                    warnings.warn("Using 1F1B scheduler with pp_degree == 1.")
                tasks, task_id_to_rank = run1f1b(
                    program,
                    cur_rank,
                    fleet_opt.get('num_micro_batches', 1),
                    fleet_opt.get('dist_strategy', {}),
                    nrank,
                    with_standalone_executor,
                )
            elif scheduler == 'Origin':
                from paddle.distributed.fleet.fleet_executor_utils import origin

                if (
                    "dist_strategy" in fleet_opt
                    and "pp_degree" in fleet_opt["dist_strategy"]
                ):
                    assert (
                        fleet_opt["dist_strategy"]["pp_degree"] == 1
                    ), "For pipeline mode, the scheduler should be 1F1B instead of Origin."
                if "num_micro_batches" in fleet_opt:
                    assert (
                        fleet_opt["num_micro_batches"] == 1
                    ), "For origin scheduler mode, the num micro batches should be 1."
                tasks, task_id_to_rank = origin(program, cur_rank)
            else:
                raise "Fleet_executor only supports 1F1B and Origin scheduler, " "but received " + str(
                    scheduler
                ) + "."
            # NOTE: have to hold these vars, otherwise will be destructed
            fleet_opt['tasks'] = tasks
            fleet_opt['task_id_to_rank'] = task_id_to_rank
        place = core.Place()
        place.set_place(self.place)

        inference_root_scope_vars = (
            fleet_opt["fetch_var"] if "fetch_var" in fleet_opt else []
        )
        self._fleet_executor.init(
            carrier_id,
            program.desc,
            scope,
            place,
            num_micro_batches,
            tasks,
            task_id_to_rank,
            inference_root_scope_vars,
            micro_scope_list,
        )

    def _run_using_fleet_executor(
        self,
        program=None,
        feed=None,
        feed_var_name="feed",
        fetch_var_name="fetch",
        fetch_list=None,
        with_standalone_executor=False,
        return_numpy=True,
    ):
        cache_key = _get_strong_program_cache_key(program, feed, fetch_list)
        cached_program = self._get_program_cache(cache_key)
        cached_scope = self._get_scope_cache(cache_key)
        micro_cached_scopes = self._get_micro_scopes_cache(cache_key)
        fleet_opt = program._pipeline_opt["fleet_opt"]
        if cached_scope is None:
            cached_scope = global_scope()
            self._add_scope_cache(cache_key, cached_scope)
        if micro_cached_scopes is None:
            micro_cached_scopes = []
            if (
                "inference_generation" in fleet_opt
                and fleet_opt["inference_generation"]
            ):
                for _ in range(int(fleet_opt["num_micro_batches"])):
                    micro_cached_scopes.append(cached_scope.new_scope())
                self._add_micro_scopes_cache(cache_key, micro_cached_scopes)
        if cached_program is None:
            assert (
                program._pipeline_opt
            ), "program should have _pipeline_opt to start carrier"
            real_feed = [] if feed is None else feed
            real_program = program
            if "section_program" in program._pipeline_opt:
                real_program = program._pipeline_opt["section_program"]
            cached_program = _add_feed_fetch_ops(
                program=real_program,
                feed=real_feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name,
            )
            main_block = cached_program.block(0)
            for op in main_block.ops:
                # set the op_role of fetch op to Optimize to avoid
                # erase the fetched vars by gc for pipeline
                if op.type == 'fetch':
                    op._set_attr(
                        'op_role',
                        core.op_proto_and_checker_maker.OpRole.Optimize,
                    )
            self._add_program_cache(cache_key, cached_program)
            fleet_opt = program._pipeline_opt["fleet_opt"]
            if 'tasks' in fleet_opt:
                # Insert feed/fetch op for cloned program in each task node,
                # these ops has already been inserted into the origin program.
                # To avoid every task nodes all have feed/fetch ops,
                # only insert feed ops into the first task node,
                # then insert fetch ops into the last task node.

                # Insert feed ops
                feed_task = fleet_opt['tasks'][0]
                print("Inserting feed ops for task", feed_task.task_id())
                feed_program = feed_task.get_program()
                feed_program = self._add_feed_ops(
                    program=feed_program,
                    feed=real_feed,
                    feed_var_name=feed_var_name,
                )
                feed_task.set_program(feed_program)

                # Insert fetch ops
                fetch_task = fleet_opt['tasks'][-1]
                print("Inserting fetch ops for task", fetch_task.task_id())
                fetch_program = fetch_task.get_program()
                fetch_program = self._add_fetch_ops(
                    program=fetch_program,
                    fetch_list=fetch_list,
                    fetch_var_name=fetch_var_name,
                )
                main_block = fetch_program.block(0)
                for op in main_block.ops:
                    # set the op_role of fetch op to Optimize to avoid
                    # erase the fetched vars by gc for pipeline
                    if op.type == 'fetch':
                        op._set_attr(
                            'op_role',
                            core.op_proto_and_checker_maker.OpRole.Optimize,
                        )
                fetch_task.set_program(fetch_program)

            micro_scope_list = []
            if (
                "inference_generation" in fleet_opt
                and fleet_opt["inference_generation"]
            ):
                for i in range(int(fleet_opt["num_micro_batches"])):
                    micro_scope_list.append(cached_scope.new_scope())

            self._prepare_fleet_executor_carrier(
                cache_key,
                program=cached_program,
                scope=cached_scope,
                fleet_opt=fleet_opt,
                micro_scope_list=micro_cached_scopes,
                with_standalone_executor=with_standalone_executor,
            )

        if feed:
            # NOTE: don't have to traverse programs in task nodes,
            # since they all sub program of cached program and
            # cached program is also added feed fetch var
            self._feed_data(cached_program, feed, feed_var_name, cached_scope)

        from paddle.optimizer.lr import LRScheduler

        if hasattr(program, 'lr_scheduler'):
            lr_scheduler = program.lr_scheduler
            assert isinstance(lr_scheduler, LRScheduler), "must be LRScheduler"
            lr_value = lr_scheduler()
            lr_var = program.global_block().vars[lr_scheduler._var_name]
            data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
            tensor = core.get_variable_tensor(
                cached_scope, lr_scheduler._var_name
            )
            tensor.set(data, self.place)

        self._fleet_executor.run(cache_key)
        if "fetch_var" in fleet_opt:
            # If we speed up the generation in evaluation, we need to generate
            # multiple queries at the same time. Each query will in separate scope in order
            # not mix up. It indicate that final result will in multiple scopes and need to
            # fetch each.
            result_list = []
            for scope in micro_cached_scopes:
                scope_result_list = []
                for varname in fleet_opt["fetch_var"]:
                    tensor = None
                    try:
                        tensor = core.get_variable_tensor(scope, varname)
                        if return_numpy:
                            tensor = as_numpy(tensor)
                    except:
                        var = scope.find_var(varname)
                        tensor = var.get_lod_tensor_array()
                        if return_numpy:
                            tensor = as_numpy(tensor)
                        else:
                            tensor = list(tensor)

                    if tensor:
                        scope_result_list.append(tensor)

                if scope_result_list:
                    result_list.append(scope_result_list)
            return result_list

        if fetch_list:
            arr = cached_scope.find_var(fetch_var_name).get_fetch_list()
            tensors = arr._move_to_list()
            return as_numpy(tensors)
        return None

    def _add_feed_ops(self, program, feed, feed_var_name):
        tmp_program = program.clone()

        global_block = tmp_program.global_block()

        if feed_var_name in global_block.vars:
            feed_var = global_block.var(feed_var_name)
        else:
            feed_var = global_block.create_var(
                name=feed_var_name,
                type=core.VarDesc.VarType.FEED_MINIBATCH,
                persistable=True,
            )

        # prepend feed operators
        if not has_feed_operators(global_block, feed, feed_var_name):
            for i, name in enumerate(feed):
                if global_block.has_var(name):
                    out = global_block.var(name)
                    global_block._prepend_op(
                        type='feed',
                        inputs={'X': [feed_var]},
                        outputs={'Out': [out]},
                        attrs={'col': i},
                    )
                else:
                    warnings.warn(
                        "The variable %s is not found in program. It is not declared or is pruned."
                        % name
                    )

        return tmp_program

    @classmethod
    def _add_fetch_ops(
        cls, program, fetch_list, fetch_var_name, use_fetch_v2=False
    ):
        tmp_program = program.clone()

        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True,
            )

        if use_fetch_v2:
            fetch_op = 'fetch_v2'
        else:
            fetch_op = 'fetch'

        # append fetch_operators
        if not has_fetch_operators(
            global_block, fetch_list, fetch_var_name, fetch_op
        ):
            for i, var in enumerate(fetch_list):
                assert isinstance(
                    var, (Variable, str)
                ), f"Wrong type for fetch_list[{i}]: {type(var)}"
                global_block.append_op(
                    type=fetch_op,
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i},
                )

        return tmp_program

    @classmethod
    def _remove_fetch_ops(cls, program, fetch_op_name='fetch'):
        tmp_program = program.clone()
        global_block = tmp_program.global_block()
        op_num = len(global_block.ops)
        for idx in reversed(range(op_num)):
            if global_block.ops[idx].type == fetch_op_name:
                global_block._remove_op(idx)

        return tmp_program

    def _run_pipeline(
        self,
        program=None,
        dataset=None,
        scope=None,
        thread=0,
        is_infer=False,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
        fetch_handler=None,
        use_program_cache=False,
    ):
        scope, real_fetch_list, trainer_instance = self._prepare_pipeline_ctx(
            program,
            dataset,
            scope,
            thread,
            is_infer,
            debug,
            fetch_list,
            fetch_info,
            print_period,
            fetch_handler,
            use_program_cache,
        )

        from paddle.optimizer.lr import LRScheduler

        if hasattr(program, 'lr_scheduler'):
            lr_scheduler = program.lr_scheduler
            assert isinstance(lr_scheduler, LRScheduler), "must be LRScheduler"
            lr_value = lr_scheduler()
            lr_var = program.global_block().vars[lr_scheduler._var_name]
            data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
            tensor = core.get_variable_tensor(scope, lr_scheduler._var_name)
            tensor.set(data, self.place)

        self._default_executor.run_from_dataset(trainer_instance)

        if not use_program_cache:
            self._default_executor.release_trainer(trainer_instance)

        if real_fetch_list:
            arr = scope.find_var('fetch').get_fetch_list()
            tensors = arr._move_to_list()
            return as_numpy(tensors)

        return None

    def infer_from_dataset(
        self,
        program=None,
        dataset=None,
        scope=None,
        thread=0,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
        fetch_handler=None,
    ):
        """
        Infer from a pre-defined Dataset. Dataset is defined in paddle.base.dataset.
        Given a program, either a program or compiled program, infer_from_dataset will
        consume all data samples in dataset. Input scope can be given by users. By default,
        scope is global_scope(). The total number of thread run in training is `thread`.
        Thread number used in training will be minimum value of threadnum in Dataset and
        the value of thread in this interface. Debug can be set so that executor will display
        Run-Time for all operators and the throughputs of current infer task.

        The document of infer_from_dataset is almost the same as train_from_dataset,
        except that in distributed training, push gradients will be disabled in infer_from_dataset.
        infer_from_dataset() can be used for evaluation in multi-threadvery easily.

        Args:
            program(Program|CompiledProgram): the program that needs to be run,
                if not provided, then default_main_program (not compiled) will be used.
            dataset(paddle.base.Dataset): dataset created outside this function,
                a user should provide a well-defined dataset before calling this function.
                Please check the document of Dataset if needed. default is None
            scope(Scope): the scope used to run this program, you can switch it to different scope
                for each run. default is global_scope
            thread(int): number of thread a user wants to run in this function. Default is 0, which
                means using thread num of dataset
            debug(bool): whether a user wants to run infer_from_dataset, default is False
            fetch_list(Tensor List): fetch Tensor list, each Tensor will be printed during
                training, default is None
            fetch_info(String List): print information for each Tensor, default is None
            print_period(int): the number of mini-batches for each print, default is 100
            fetch_handler(FetchHandler): a user define class for fetch output.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()
                >>> place = paddle.CPUPlace()  # you can set place = paddle.CUDAPlace(0) to use gpu
                >>> exe = paddle.static.Executor(place)
                >>> x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
                >>> y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
                >>> dataset = paddle.base.DatasetFactory().create_dataset()
                >>> dataset.set_use_var([x, y])
                >>> dataset.set_thread(1)
                >>> # you should set your own filelist, e.g. filelist = ["dataA.txt"]
                >>> filelist = []
                >>> dataset.set_filelist(filelist)
                >>> exe.run(paddle.static.default_startup_program())
                >>> exe.infer_from_dataset(program=paddle.static.default_main_program(),
                ...                         dataset=dataset)
        """
        return self._run_from_dataset(
            program,
            dataset,
            scope,
            thread,
            True,
            debug,
            fetch_list,
            fetch_info,
            print_period,
            fetch_handler,
        )

    def start_heter_trainer(
        self,
        program=None,
        scope=None,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
        fetch_handler=None,
    ):
        scope, trainer = self._prepare_trainer(
            program=program,
            dataset=None,
            scope=scope,
            thread=1,
            debug=debug,
            fetch_list=fetch_list,
            fetch_info=fetch_info,
            print_period=print_period,
        )

        trainer._set_infer(False)
        trainer._gen_trainer_desc()

        self._dump_debug_info(program=program, trainer=trainer)

        trainer_instance = self._default_executor.init_for_dataset(
            program.desc, trainer._desc(), scope, None
        )

        # if fetch_handler is not None:
        #    scope0 = trainer_instance.get_worker_scope(0)
        #    fetch_monitor = FetchHandlerMonitor(scope0, fetch_handler)
        #    fetch_monitor.start()
        #    self._default_executor.run_from_dataset(trainer_instance)
        #    fetch_monitor.stop()
        #    self._default_executor.release_trainer(trainer_instance)
        # else:

        self._default_executor.run_from_dataset(trainer_instance)
        # self._default_executor.release_trainer(trainer_instance)

        return trainer_instance

    def train_from_dataset(
        self,
        program=None,
        dataset=None,
        scope=None,
        thread=0,
        debug=False,
        fetch_list=None,
        fetch_info=None,
        print_period=100,
        fetch_handler=None,
    ):
        """
        Train from a pre-defined Dataset. Dataset is defined in paddle.base.dataset.
        Given a program, either a program or compiled program, train_from_dataset will
        consume all data samples in dataset. Input scope can be given by users. By default,
        scope is global_scope(). The total number of thread run in training is `thread`.
        Thread number used in training will be minimum value of threadnum in Dataset and
        the value of thread in this interface. Debug can be set so that executor will display
        Run-Time for all operators and the throughputs of current training task.

        Note: train_from_dataset will destroy all resources created within executor for each run.

        Args:
            program(Program|CompiledProgram): the program that needs to be run,
                if not provided, then default_main_program (not compiled) will be used.
            dataset(paddle.base.Dataset): dataset created outside this function,
                a user should provide a well-defined dataset before calling this function.
                Please check the document of Dataset if needed.
            scope(Scope): the scope used to run this program, you can switch it to different scope
                for each run. default is global_scope
            thread(int): number of thread a user wants to run in this function. Default is 0, which
                means using thread num of dataset
            debug(bool): whether a user wants to run train_from_dataset
            fetch_list(Tensor List): fetch Tensor list, each variable will be printed
                during training
            fetch_info(String List): print information for each Tensor, its length should be equal
                to fetch_list
            print_period(int): the number of mini-batches for each print, default is 100
            fetch_handler(FetchHandler): a user define class for fetch output.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()
                >>> place = paddle.CPUPlace() # you can set place = paddle.CUDAPlace(0) to use gpu
                >>> exe = paddle.static.Executor(place)
                >>> x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
                >>> y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
                >>> dataset = paddle.base.DatasetFactory().create_dataset()
                >>> dataset.set_use_var([x, y])
                >>> dataset.set_thread(1)
                >>> # you should set your own filelist, e.g. filelist = ["dataA.txt"]
                >>> filelist = []
                >>> dataset.set_filelist(filelist)
                >>> exe.run(paddle.static.default_startup_program())
                >>> exe.train_from_dataset(program=paddle.static.default_main_program(),
                ...                         dataset=dataset)
        """
        return self._run_from_dataset(
            program,
            dataset,
            scope,
            thread,
            False,
            debug,
            fetch_list,
            fetch_info,
            print_period,
            fetch_handler,
        )
