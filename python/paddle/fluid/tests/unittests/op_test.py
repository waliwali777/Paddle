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

import os
import unittest
import warnings
import numpy as np
import random
import six
import time
import itertools
import collections
from collections import defaultdict

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder, Variable
from testsuite import create_op, set_input, append_input_output, append_loss_ops
from paddle.fluid import unique_name
from white_list import op_accuracy_white_list, op_check_grad_white_list, compile_vs_runtime_white_list


def _set_use_system_allocator(value=None):
    USE_SYSTEM_ALLOCATOR_FLAG = "FLAGS_use_system_allocator"
    old_value = core.globals()[USE_SYSTEM_ALLOCATOR_FLAG]
    value = old_value if value is None else value
    core.globals()[USE_SYSTEM_ALLOCATOR_FLAG] = value
    return old_value


def randomize_probability(batch_size, class_num, dtype='float32'):
    prob = np.random.uniform(
        0.1, 1.0, size=(batch_size, class_num)).astype(dtype)
    prob_sum = prob.sum(axis=1)
    for i in six.moves.xrange(len(prob)):
        prob[i] /= prob_sum[i]
    return prob


def get_numeric_gradient(place,
                         scope,
                         op,
                         inputs,
                         input_to_check,
                         output_names,
                         delta=0.005,
                         in_place=False):
    # FIXME: change this method by compile time concepts
    set_input(scope, op, inputs, place)

    def product(dim):
        return six.moves.reduce(lambda a, b: a * b, dim, 1)

    tensor_to_check = scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.shape())
    tensor_to_check_dtype = tensor_to_check._dtype()
    if tensor_to_check_dtype == core.VarDesc.VarType.FP32:
        tensor_to_check_dtype = np.float32
    elif tensor_to_check_dtype == core.VarDesc.VarType.FP64:
        tensor_to_check_dtype = np.float64
    elif tensor_to_check_dtype == core.VarDesc.VarType.FP16:
        tensor_to_check_dtype = np.float16
        # set delta as np.float16, will automatic convert to float32, float64
        delta = np.array(delta).astype(np.float16)
    else:
        raise ValueError("Not supported data type " + str(
            tensor_to_check_dtype))

    def get_output():
        sum = []
        op.run(scope, place)
        for output_name in output_names:
            sum.append(
                np.array(scope.find_var(output_name).get_tensor()).astype(
                    tensor_to_check_dtype).mean())
        return tensor_to_check_dtype(np.array(sum).sum() / len(output_names))

    gradient_flat = np.zeros(shape=(tensor_size, ), dtype=tensor_to_check_dtype)

    def __get_elem__(tensor, i):
        if tensor_to_check_dtype == np.float16:
            numpy_tensor = np.array(tensor).astype(np.float16)
            numpy_tensor = numpy_tensor.flatten()
            return numpy_tensor[i]
        elif tensor_to_check_dtype == np.float32:
            return tensor._get_float_element(i)
        else:
            return tensor._get_double_element(i)

    def __set_elem__(tensor, i, e):
        if tensor_to_check_dtype == np.float16:
            numpy_tensor = np.array(tensor).astype(np.float16)
            shape = numpy_tensor.shape
            numpy_tensor = numpy_tensor.flatten()
            numpy_tensor[i] = e
            numpy_tensor = numpy_tensor.reshape(shape)
            tensor.set(numpy_tensor, place)
        elif tensor_to_check_dtype == np.float32:
            tensor._set_float_element(i, e)
        else:
            tensor._set_double_element(i, e)

    # we only compute gradient of one element each time.
    # we use a for loop to compute the gradient of every element.
    for i in six.moves.xrange(tensor_size):
        if in_place:
            set_input(scope, op, inputs, place)

        # get one input element throw it's index i.
        origin = __get_elem__(tensor_to_check, i)
        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        __set_elem__(tensor_to_check, i, x_pos)
        y_pos = get_output()

        if in_place:
            set_input(scope, op, inputs, place)

        x_neg = origin - delta
        __set_elem__(tensor_to_check, i, x_neg)
        y_neg = get_output()

        __set_elem__(tensor_to_check, i, origin)
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    return gradient_flat.reshape(tensor_to_check.shape())


class OpTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls.call_once = False
        cls.dtype = "float32"
        cls.outputs = {}

        np.random.seed(123)
        random.seed(124)

        cls._use_system_allocator = _set_use_system_allocator(True)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        _set_use_system_allocator(cls._use_system_allocator)

    def try_call_once(self, data_type):
        if not self.call_once:
            self.call_once = True
            self.dtype = data_type

    def infer_dtype_from_inputs_outputs(self, inputs, outputs):
        def is_np_data(input):
            return isinstance(input, (np.ndarray, np.generic))

        def infer_dtype(numpy_dict, dtype_set):
            assert isinstance(
                numpy_dict,
                dict), "self.inputs, self.outputs must be numpy_dict"
            # the inputs are as follows:
            # case 1: inputs = {'X': x}
            # case 2: inputs = {'X': (x, x_lod)}
            # case 3: inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
            # case 4: inputs = {'X': [("x1", (x1, [x1_lod1])), ("x2", (x2, [x2_.lod2]))]}
            # TODO(juncaipeng) infer dtype from inputs maybe obtain wrong type.
            for _, var_value in six.iteritems(numpy_dict):
                if is_np_data(var_value):  # case 1
                    dtype_set.add(var_value.dtype)
                elif isinstance(var_value, (list, tuple)):  # case 2, 3, 4
                    for sub_val_value in var_value:
                        if is_np_data(sub_val_value):  # case 2
                            dtype_set.add(sub_val_value.dtype)
                        elif len(sub_val_value) > 1 and is_np_data(
                                sub_val_value[1]):  # case 3
                            dtype_set.add(sub_val_value[1].dtype)
                        elif len(sub_val_value) > 1 and isinstance(sub_val_value[1], (list, tuple)) \
                            and is_np_data(sub_val_value[1][0]): # case 4
                            dtype_set.add(sub_val_value[1][0].dtype)

        # infer dtype from inputs, and dtype means the precision of the test
        # collect dtype of all inputs
        dtype_set = set()
        infer_dtype(inputs, dtype_set)
        dtype_list = [
            np.dtype(np.float64), np.dtype(np.float32), np.dtype(np.float16),
            np.dtype(np.int64), np.dtype(np.int32), np.dtype(np.int16),
            np.dtype(np.int8)
        ]
        # check the dtype in dtype_list in order, select the first dtype that in dtype_set
        for dtype in dtype_list:
            if dtype in dtype_set:
                self.dtype = dtype
                break

    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], place)
                        tensor.set_recursive_sequence_lengths(np_value[1])
                    else:
                        tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    tensor.set_recursive_sequence_lengths(self.inputs[var_name][
                        1])
                else:
                    tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def _append_ops(self, block):
        self.__class__.op_type = self.op_type  # for ci check, please not delete it for now
        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
        "infer datatype from inputs and outputs for this test case"
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        inputs = append_input_output(block, op_proto, self.inputs, True,
                                     self.dtype)
        outputs = append_input_output(block, op_proto, self.outputs, False,
                                      self.dtype)

        if hasattr(self, "cache_name_list"):
            for name in self.cache_name_list:
                inputs[name] = block.create_var(
                    name=name,
                    persistable=True,
                    type=core.VarDesc.VarType.RAW,
                    stop_gradient=True)

        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=self.attrs if hasattr(self, "attrs") else dict())
        # infer variable type and infer shape in compile-time 
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        return op

    def _get_io_vars(self, block, numpy_inputs):
        inputs = {}
        for name, value in six.iteritems(numpy_inputs):
            if isinstance(value, list):
                var_list = [
                    block.var(sub_name) for sub_name, sub_value in value
                ]
                inputs[name] = var_list
            else:
                inputs[name] = block.var(name)
        return inputs

    def _get_inputs(self, block):
        return self._get_io_vars(block, self.inputs)

    def _get_outputs(self, block):
        return self._get_io_vars(block, self.outputs)

    def calc_output(self, place):
        outs, _ = self._calc_output(place)
        return outs

    def _create_var_from_numpy(self, value):
        if isinstance(value, tuple):
            data = value[0]
            lod = value[1]
            v = fluid.dygraph.base.to_variable(value=data)
            v.value().get_tensor().set_recursive_sequence_lengths(lod)
            return v
        else:
            return fluid.dygraph.base.to_variable(value)

    def append_input_output_for_dygraph(self, op_proto, np_list, is_input,
                                        if_return_inputs_grad_dict, block):
        def create_var(np_value, name, is_input, if_return_inputs_grad_dict):
            np_value_temp = np_value
            has_lod = False
            lod_temp = None
            if isinstance(np_value, tuple):
                np_value_temp = np_value[0]
                has_lod = True
                lod_temp = np_value[1]

            if is_input:
                v = self._create_var_from_numpy(np_value_temp)
                if if_return_inputs_grad_dict:
                    v.stop_gradient = False
                if has_lod:
                    v.value().get_tensor().set_recursive_sequence_lengths(
                        lod_temp)
            else:
                v = block.create_var(
                    name=name,
                    dtype=np_value_temp.dtype,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
                    stop_gradient=False)

            return v

        # prepare variable for input or output
        var_dict = defaultdict(list)
        if if_return_inputs_grad_dict:
            inputs_grad_dict = defaultdict()
        proto_list = op_proto.inputs if is_input else op_proto.outputs
        for var_proto in proto_list:
            name = var_proto.name
            if (name not in np_list) and var_proto.dispensable:
                continue
            if name not in np_list:
                assert var_proto.intermediate, "{} not found".format(name)
                v = block.create_var(
                    dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR)
                var_dict[name].append(v)
                if if_return_inputs_grad_dict:
                    inputs_grad_dict[name] = v
                continue
            if var_proto.duplicable:
                assert isinstance(
                    np_list[name],
                    list), "Duplicable {} should be set as list".format(name)
                var_list = []
                slot_name = name
                for (name, np_value) in np_list[name]:
                    v = create_var(np_value, name, is_input,
                                   if_return_inputs_grad_dict)
                    var_list.append(v)
                    if if_return_inputs_grad_dict:
                        inputs_grad_dict[name] = v
                var_dict[slot_name] = var_list
            else:
                nplist_value_temp = None
                name_temp = None
                if isinstance(np_list[name], list):
                    nplist_value_temp = np_list[name][0]
                    name_temp = name
                else:
                    nplist_value_temp = np_list[name]
                    name_temp = unique_name.generate("%s_out" % (name))
                v = create_var(nplist_value_temp, name_temp, is_input,
                               if_return_inputs_grad_dict)
                var_dict[name].append(v)
                if if_return_inputs_grad_dict:
                    inputs_grad_dict[name] = v

        if if_return_inputs_grad_dict:
            return var_dict, inputs_grad_dict
        else:
            return var_dict

    def _calc_dygraph_output(self, place, parallel=False, no_check_set=None):
        self.__class__.op_type = self.op_type  # for ci check, please not delete it for now
        with fluid.dygraph.base.guard(place=place):
            block = fluid.default_main_program().global_block()

            op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

            # prepare input variable
            inputs = self.append_input_output_for_dygraph(op_proto, self.inputs,
                                                          True, False, block)

            # prepare output variable
            outputs = self.append_input_output_for_dygraph(
                op_proto, self.outputs, False, False, block)

            # prepare attrbutes
            attrs_outputs = {}
            if hasattr(self, "attrs"):
                for attrs_name in self.attrs:
                    if self.attrs[attrs_name] is not None:
                        attrs_outputs[attrs_name] = self.attrs[attrs_name]
            block.append_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs_outputs if hasattr(self, "attrs") else None)
            return outputs

    def _calc_output(self,
                     place,
                     parallel=False,
                     no_check_set=None,
                     loss=None,
                     enable_inplace=None,
                     for_inplace_test=None):
        program = Program()
        block = program.global_block()
        op = self._append_ops(block)

        inputs = self._get_inputs(block)
        outputs = self._get_outputs(block)
        feed_map = self.feed_var(inputs, place)

        if for_inplace_test:
            # Some variables' tensors hold no buffer (tensor's _holder is NULL), like XShape in reshape2 op, 
            # and the shapes of those variables contain 0 (eg. Xshape.shape = [0, 2, 5]). 
            # Set persistable for those variables in order to get them from global_scope for inplace grad test directly other than feed them,
            # since feed op calls check_memory_size() which fails when tensor's holder_ is NULL.
            for out_name in op.output_arg_names:
                var = block.var(out_name)
                if 0 in var.shape:
                    var.persistable = True
        original_program = program
        if parallel:
            use_cuda = False
            if isinstance(place, fluid.CUDAPlace):
                use_cuda = True
            compiled_prog = fluid.CompiledProgram(program).with_data_parallel(
                loss_name=loss.name if loss else None, places=place)
            program = compiled_prog
        fetch_list = getattr(self, "fetch_list", [])
        # if the fetch_list is customized by user, we use it directly.
        # if not, fill the fetch_list by the user configured outputs in test.
        if len(fetch_list) == 0:
            for var_name, var in six.iteritems(outputs):
                if no_check_set is not None and var_name in no_check_set:
                    continue
                if isinstance(var, list):
                    for v in var:
                        fetch_list.append(v.name)
                else:
                    fetch_list.append(var.name)
        # if the fetch_list still empty, fill the fetch_list by the operator output.
        if len(fetch_list) == 0:
            for out_name, out_dup in Operator.get_op_outputs(self.op_type):
                fetch_list.append(str(out_name))

        if enable_inplace is not None:
            build_strategy = fluid.BuildStrategy()
            build_strategy.enable_inplace = enable_inplace

            compiled_prog = fluid.CompiledProgram(program).with_data_parallel(
                build_strategy=build_strategy, places=place)
            program = compiled_prog

        executor = Executor(place)
        outs = executor.run(program,
                            feed=feed_map,
                            fetch_list=fetch_list,
                            return_numpy=False)
        self.op = op
        self.program = original_program
        if for_inplace_test:
            return outs, fetch_list, feed_map, original_program, op.desc
        else:
            return outs, fetch_list

    def _compare_expect_and_actual_outputs(self,
                                           place,
                                           fetch_list,
                                           expect_outs,
                                           actual_outs,
                                           inplace_atol=None):
        """Compare expect outs and actual outs of an tested op.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs. 
            fetch_list (list): The outputs of tested op.
            expect_outs (list): The expect outs of tested op.
            actual_outs (list): The actual outs of tested op.
            inplace_atol (float): The tolerable error, only set when tested op doesn't ensure computational consistency, like group_norm op.

        Returns:
            None.
        """
        # compare expect_outs and actual_outs
        for i, name in enumerate(fetch_list):
            if inplace_atol is not None:
                self.assertTrue(
                    np.allclose(
                        np.array(expect_outs[i]),
                        np.array(actual_outs[i]),
                        atol=inplace_atol),
                    "Output (" + name + ") has diff at " + str(place) +
                    " when using and not using inplace" + "\nExpect " +
                    str(expect_outs[i]) + "\n" + "But Got" + str(actual_outs[i])
                    + " in class " + self.__class__.__name__)
            else:
                self.assertTrue(
                    np.array_equal(
                        np.array(expect_outs[i]), np.array(actual_outs[i])),
                    "Output (" + name + ") has diff at " + str(place) +
                    " when using and not using inplace" + "\nExpect " +
                    str(expect_outs[i]) + "\n" + "But Got" + str(actual_outs[i])
                    + " in class " + self.__class__.__name__ + '\n')

    def _construct_grad_program_from_forward(self, fwd_program, grad_op_desc,
                                             op_grad_to_var):
        """Generate grad_program which contains the grad_op.

        Args:
            fwd_program (tuple): The program that contains grad_op_desc's corresponding forward op.
            grad_op_desc (OpDesc): The OpDesc of grad op.
            op_grad_to_var (dict): The relation of variables in grad op and its forward op. 

        Returns:
            grad_program (program): The program which contains the grad_op.
        """
        grad_program = Program()
        grad_block = grad_program.global_block()
        new_op_desc = grad_block.desc.append_op()
        new_op_desc.copy_from(grad_op_desc)
        grad_program._sync_with_cpp()

        # Create grad vars based on fwd vars (shape and dtype)
        for arg in grad_op_desc.input_arg_names(
        ) + grad_op_desc.output_arg_names():
            fwd_var_name = op_grad_to_var.get(arg, None)
            if fwd_var_name is None:
                fwd_var_name = arg
            fwd_var = fwd_program.global_block().vars.get(fwd_var_name)
            assert fwd_var is not None, "{} cannot be found".format(
                fwd_var_name)
            grad_var = grad_block.create_var(
                name=arg,
                dtype=fwd_var.dtype,
                shape=fwd_var.shape,
                type=fwd_var.type,
                persistable=False)

            # Some variables' tensors hold no buffer (tensor's _holder is NULL), like XShape in reshape2 op, 
            # and the shapes of those variables contain 0 (eg. Xshape.shape = [0, 2, 5]). 
            # Set persistable for those variables in order to get them from global_scope for inplace grad test directly other than feed them,
            # since feed op calls check_memory_size() which fails when tensor's holder_ is NULL.
            if 0 in grad_var.shape:
                grad_var.persistable = True
        grad_program._sync_with_cpp()
        return grad_program

    def _construct_grad_feed_map_from_forward(self, place, fwd_res,
                                              grad_op_desc, op_grad_to_var):
        """Generate grad_feed_map for grad_program.

        since we don`t really check gradient accuracy, but check the consistency when using and not using inplace,
        we use fwd outs (also inputs sometimes) to construct grad inputs.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs. 
            fwd_res (tuple): The outputs of its forward op, in the same form as returns of _calc_outputs() when for_inplace_test is True.
                i.e., tuple(fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc)
            grad_op_desc (OpDesc): The OpDesc of grad op.
            op_grad_to_var (dict): The relation of variables in grad op and its fwd_op. 

        Returns:
            grad_feed_map (dict): The feed_map of grad_op.
        """
        fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc = fwd_res
        p = core.Place()
        p.set_place(place)
        grad_feed_map = {}
        for arg in grad_op_desc.input_arg_names():
            if arg in fwd_feed_map.keys():
                grad_feed_map[arg] = fwd_feed_map[arg]._copy(p)
            else:
                fwd_var_name = op_grad_to_var.get(arg, None)
                if fwd_var_name is None:
                    fwd_var_name = arg

                for i, out_name in enumerate(fwd_fetch_list):
                    if out_name == fwd_var_name:
                        # don't feed variables whose tensors hold no buffer (shape contains 0 like shape = [0,2,5] and holder_ is NULL), like XShape in reshape2 op.
                        # get them from global_scope directly since we have set them persistable in fwd execution
                        if 0 in fwd_program.global_block().var(out_name).shape:
                            continue
                        else:
                            grad_feed_map[arg] = fwd_outs[i]._copy(p)
        return grad_feed_map

    def _get_need_run_ops(self, op_desc, fwd_op_desc=None):
        """Postorder traversal of the 'grad' tree to get all ops that need to run during inplace test.
        An op needs to run druing inplace check if,
        (1) it has infer_inplace,
        (2) it has infer_inplace in its grad descendants. (since we need its outputs as to construct its grad's inputs)
        
        Args:
            op_desc (OpDesc): The op_desc of current op. 
            fwd_op_desc (OpDesc): The op_desc of current op's forward op, None if current op has no forward op. 
                Eg. relu's fwd_op is None, relu_grad's fwd_op is relu, relu_grad_grad's fwd_op is relu_grad, etc.
            
        Returns:
            need_run_ops (list[(op_desc, fwd_op_desc)]): The ops that need to run during inplace test.
        """
        need_run_ops = []
        visited_ops = []

        def _dfs_grad_op(op_desc, fwd_op_desc=None):
            visited_ops.append(op_desc.type())
            has_infer_inplace = fluid.core.has_infer_inplace(op_desc.type())
            has_grad_op_maker = fluid.core.has_grad_op_maker(op_desc.type())
            has_infer_inplace_in_grad_descendants = False
            if not has_grad_op_maker:
                has_infer_inplace_in_descendants = False
            else:
                # get grad_op_desc 
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    op_desc, set(), [])
                if not grad_op_desc_list:
                    has_infer_inplace_in_grad_descendants = False
                else:
                    for i, grad_op_desc in enumerate(grad_op_desc_list):
                        if grad_op_desc.type(
                        ) not in visited_ops and _dfs_grad_op(
                                grad_op_desc, fwd_op_desc=op_desc):
                            has_infer_inplace_in_grad_descendants = True
            if has_infer_inplace or has_infer_inplace_in_grad_descendants:
                need_run_ops.append((op_desc, fwd_op_desc))
                return True
            else:
                return False

        _dfs_grad_op(op_desc, fwd_op_desc=fwd_op_desc)
        return need_run_ops

    def _check_forward_inplace(self,
                               place,
                               no_check_set=None,
                               inplace_atol=None):
        """Chech the inplace correctness of given op (self.op_type).
        Run the op twice with same inputs, one enable inplace and another disable, compare their outputs.
        
        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs. 
            no_check_set (list): The names of outputs that needn't check, like XShape of reshape op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            expect_res (tuple(outs, fetch_list, feed_map, program, op_desc)): The results of given op. 
                We return this to construct grad_program and grad_feed_map for grad inplace check. 
        """
        # _calc_output() returns in the form tuple(outs, fetch_list, feed_map, program, op_desc) when for_inplace_test=True.
        expect_res = self._calc_output(
            place,
            no_check_set=no_check_set,
            enable_inplace=False,
            for_inplace_test=True)
        actual_res = self._calc_output(
            place,
            no_check_set=no_check_set,
            enable_inplace=True,
            for_inplace_test=True)
        # compare expect_outs and actual_outs
        self._compare_expect_and_actual_outputs(
            place,
            expect_res[1],
            expect_res[0],
            actual_res[0],
            inplace_atol=inplace_atol)
        return expect_res

    def _calc_grad_output(self,
                          place,
                          fwd_res,
                          grad_op_desc,
                          enable_inplace=None):
        """Calculate grad_output for given grad_op_desc.

        since we don`t really check gradient accuracy, but check the consistency when using and not using inplace,
        we use fwd outs (also inputs sometimes) to construct grad inputs.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs. 
            fwd_res (tuple): The outputs of its forward op, in the same form as returns of _calc_outputs() when for_inplace_test is True.
                i.e., tuple(fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc).
            grad_op_desc (OpDesc): The OpDesc of grad op.
            enable_inplace (bool): Enable inplace or not.

        Returns:
            res (tuple(outs, fetch_list, feed_map, program, op_desc)): The results of given grad_op_desc.
        """
        fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc = fwd_res
        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(fwd_op_desc,
                                                                  set(), [])
        grad_program = self._construct_grad_program_from_forward(
            fwd_program, grad_op_desc, op_grad_to_var)
        grad_feed_map = self._construct_grad_feed_map_from_forward(
            place, fwd_res, grad_op_desc, op_grad_to_var)
        grad_fetch_list = grad_op_desc.output_arg_names()
        exe = Executor(place)
        program = grad_program
        if enable_inplace is not None:
            build_strategy = fluid.BuildStrategy()
            build_strategy.enable_inplace = enable_inplace
            compiled_program = fluid.CompiledProgram(
                grad_program).with_data_parallel(
                    loss_name="", build_strategy=build_strategy, places=place)
            program = compiled_program
        outs = exe.run(program,
                       feed=grad_feed_map,
                       fetch_list=grad_fetch_list,
                       return_numpy=False)
        return outs, grad_fetch_list, grad_feed_map, grad_program, grad_op_desc

    def _check_grad_inplace(self,
                            place,
                            fwd_res,
                            grad_op_desc,
                            inplace_atol=None):
        """Chech the inplace correctness of given grad_op_desc.

        Run the grad op twice with same inputs, one enable inplace and another disable, compare their outputs.
        It works like _check_forward_inplace, but the way to construct program and feed_map differs.
        So we define a new function for grad, grad_grad, etc.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs. 
            fwd_res (tuple): The outputs of its forward op, in the same form as returns of _calc_outputs() when for_inplace_test is True.
                i.e., tuple(fwd_outs, fwd_fetch_list, fwd_feed_map, fwd_program, fwd_op_desc).
            grad_op_desc (OpDesc): The OpDesc of grad op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            expect_res (tuple(outs, fetch_list, feed_map, program, op_desc)): The results of given op. 
                We return this to construct grad_program and grad_feed_map for grad inplace check. 
        """
        expect_res = self._calc_grad_output(
            place, fwd_res, grad_op_desc, enable_inplace=False)
        actual_res = self._calc_grad_output(
            place, fwd_res, grad_op_desc, enable_inplace=True)
        self._compare_expect_and_actual_outputs(
            place,
            expect_res[1],
            expect_res[0],
            actual_res[0],
            inplace_atol=inplace_atol)
        return expect_res

    def check_inplace_output_with_place(self,
                                        place,
                                        no_check_set=None,
                                        inplace_atol=None):
        """Chech the inplace correctness of given op, its grad op, its grad_grad op, etc.

        (1) Get all ops need to run. (see conditions in _get_need_run_ops())
        (2) Run op in need_run_ops, and do inplace check if it has infer_inplace.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs. 
            no_check_set (list): The names of outputs that needn't check, like XShape of reshape op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            None
        """
        has_infer_inplace = fluid.core.has_infer_inplace(self.op_type)
        has_grad_op_maker = fluid.core.has_grad_op_maker(self.op_type)

        fwd_res = self._calc_output(
            place, no_check_set=no_check_set, for_inplace_test=True)
        op_desc = fwd_res[4]
        need_run_ops = self._get_need_run_ops(op_desc)

        res = {}
        for op_desc, father_op_desc in reversed(need_run_ops):
            # The first one is the forward op
            has_infer_inplace = fluid.core.has_infer_inplace(op_desc.type())
            if op_desc.type() == self.op_type:
                if has_infer_inplace:
                    res[op_desc] = self._check_forward_inplace(
                        place,
                        no_check_set=no_check_set,
                        inplace_atol=inplace_atol)
                else:
                    res[op_desc] = self._calc_output(
                        place, no_check_set=no_check_set, for_inplace_test=True)
            else:
                # TODO(zhiqiu): enhance inplace_grad test for ops (sum and activation) using mkldnn/ngraph
                # skip op that use_mkldnn and use_ngraph currently
                flags_use_mkldnn = fluid.core.globals()["FLAGS_use_mkldnn"]
                attrs_use_mkldnn = hasattr(
                    self,
                    'attrs') and bool(self.attrs.get('use_mkldnn', False))
                if flags_use_mkldnn or attrs_use_mkldnn:
                    warnings.warn(
                        "check inplace_grad for ops using mkldnn is not supported"
                    )
                    continue
                use_ngraph = fluid.core.is_compiled_with_ngraph(
                ) and fluid.core.globals()["FLAGS_use_ngraph"]
                if use_ngraph:
                    warnings.warn(
                        "check inplace_grad for ops using ngraph is not supported"
                    )
                    continue
                if has_infer_inplace:
                    fwd_res = res[father_op_desc]
                    res[op_desc] = self._check_grad_inplace(
                        place, fwd_res, op_desc, inplace_atol=inplace_atol)
                else:
                    res[op_desc] = self._calc_grad_output(place, fwd_res,
                                                          op_desc)

    def check_output_with_place(self,
                                place,
                                atol,
                                no_check_set=None,
                                equal_nan=False,
                                check_dygraph=True,
                                inplace_atol=None):
        if check_dygraph:
            dygraph_outs = self._calc_dygraph_output(
                place, no_check_set=no_check_set)
        outs, fetch_list = self._calc_output(place, no_check_set=no_check_set)
        for out_name, out_dup in Operator.get_op_outputs(self.op_type):
            if out_name not in self.outputs:
                continue
            if no_check_set is not None and out_name in no_check_set:
                continue

            def find_imperative_actual(target_name, dygraph_outs, place):
                with fluid.dygraph.base.guard(place=place):
                    for name in dygraph_outs:
                        if name == target_name:
                            return dygraph_outs[name][0]
                        var_list = dygraph_outs[name]
                        for i, var in enumerate(var_list):
                            if var.name == target_name:
                                return dygraph_outs[name][i]
                    self.assertTrue(False, "Found failed {} {}".format(
                        dygraph_outs.keys(), target_name))

            def find_actual(target_name, fetch_list):
                found = [
                    i for i, var_name in enumerate(fetch_list)
                    if var_name == target_name
                ]
                self.assertTrue(
                    len(found) == 1, "Found {} {}".format(
                        len(found), target_name))
                return found[0]

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))
                for item in sub_out:
                    sub_out_name, expect = item[0], item[1]
                    if check_dygraph:
                        imperative_actual = find_imperative_actual(
                            sub_out_name, dygraph_outs, place)
                        imperative_actual_t = np.array(imperative_actual.value()
                                                       .get_tensor())
                    idx = find_actual(sub_out_name, fetch_list)
                    actual = outs[idx]
                    actual_t = np.array(actual)
                    expect_t = expect[0] \
                        if isinstance(expect, tuple) else expect
                    self.assertTrue(
                        np.allclose(
                            actual_t, expect_t, atol=atol, equal_nan=equal_nan),
                        "Output (" + sub_out_name + ") has diff at " +
                        str(place))
                    if check_dygraph:
                        self.assertTrue(
                            np.allclose(
                                imperative_actual_t,
                                expect_t,
                                atol=atol,
                                equal_nan=equal_nan),
                            "Output (" + sub_out_name + ") has diff at " +
                            str(place) + " in dygraph mode")
                    if isinstance(expect, tuple):
                        self.assertListEqual(
                            actual.recursive_sequence_lengths(), expect[1],
                            "Output (" + sub_out_name +
                            ") has different lod at " + str(place))
                        if check_dygraph:
                            self.assertListEqual(
                                imperative_actual.value().get_tensor()
                                .recursive_sequence_lengths(), expect[1],
                                "Output (" + out_name +
                                ") has different lod at " + str(place) +
                                " in dygraph mode")
            else:
                if check_dygraph:
                    imperative_actual = find_imperative_actual(
                        out_name, dygraph_outs, place)
                    imperative_actual_t = np.array(imperative_actual.value()
                                                   .get_tensor())
                idx = find_actual(out_name, fetch_list)
                actual = outs[idx]
                actual_t = np.array(actual)
                expect = self.outputs[out_name]
                expect_t = expect[0] if isinstance(expect, tuple) else expect
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, atol=atol, equal_nan=equal_nan),
                    "Output (" + out_name + ") has diff at " + str(place) +
                    "\nExpect " + str(expect_t) + "\n" + "But Got" +
                    str(actual_t) + " in class " + self.__class__.__name__)
                if check_dygraph:
                    if six.moves.reduce(
                            lambda x, y: x * y, imperative_actual_t.shape,
                            1) == 0 and six.moves.reduce(
                                lambda x, y: x * y, expect_t.shape, 1) == 0:
                        pass
                    else:
                        self.assertTrue(
                            np.allclose(
                                imperative_actual_t,
                                expect_t,
                                atol=atol,
                                equal_nan=equal_nan),
                            "Output (" + out_name + ") has diff at " +
                            str(place) + "\nExpect " + str(expect_t) + "\n" +
                            "But Got" + str(imperative_actual_t) + " in class "
                            + self.__class__.__name__)
                if isinstance(expect, tuple):
                    self.assertListEqual(actual.recursive_sequence_lengths(),
                                         expect[1], "Output (" + out_name +
                                         ") has different lod at " + str(place))
                    if check_dygraph:
                        self.assertListEqual(
                            imperative_actual.value().get_tensor()
                            .recursive_sequence_lengths(), expect[1],
                            "Output (" + out_name + ") has different lod at " +
                            str(place) + " in dygraph mode")

        # inplace_atol only used when op doesn't ensure computational consistency
        if inplace_atol is not None:
            warnings.warn(
                "By default, inplace_atol should not be set, please check it")
        # Check inplace for given op, its grad op, its grad_grad op, etc.
        # No effect on original OpTest 
        self.check_inplace_output_with_place(
            place, no_check_set=no_check_set, inplace_atol=inplace_atol)

        if check_dygraph:
            return outs, dygraph_outs, fetch_list
        else:
            return outs, fetch_list

    def check_compile_vs_runtime(self, fetch_list, fetch_outs):
        def find_fetch_index(target_name, fetch_list):
            found = [
                i for i, var_name in enumerate(fetch_list)
                if var_name == target_name
            ]
            if len(found) == 0:
                return -1
            else:
                self.assertTrue(
                    len(found) == 1,
                    "Found {} {}".format(len(found), target_name))
                return found[0]

        for name in self.op.desc.output_names():
            var_names = self.op.desc.output(name)
            for var_name in var_names:
                i = find_fetch_index(var_name, fetch_list)
                if i == -1:
                    # The output is dispensiable or intermediate.
                    break
                out = fetch_outs[i]
                if isinstance(out, core.LoDTensor):
                    lod_level_runtime = len(out.lod())
                else:
                    if isinstance(out, core.LoDTensorArray):
                        warnings.warn(
                            "The check of LoDTensorArray's lod_level is not implemented now!"
                        )
                    lod_level_runtime = 0

                var = self.program.global_block().var(var_name)
                if var.type == core.VarDesc.VarType.LOD_TENSOR:
                    lod_level_compile = var.lod_level
                else:
                    lod_level_compile = 0
                self.assertEqual(
                    lod_level_compile, lod_level_runtime,
                    "The lod_level of Output (" + name +
                    ") is different between compile-time and runtime (" +
                    str(lod_level_compile) + " vs " + str(lod_level_runtime) +
                    ")")

    def _get_places(self):
        if self.dtype == np.float16:
            if core.is_compiled_with_cuda() and core.op_support_gpu(
                    self.op_type):
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    return [place]
                else:
                    return []
            else:
                return []
        places = [fluid.CPUPlace()]
        cpu_only = self._cpu_only if hasattr(self, '_cpu_only') else False
        use_ngraph = fluid.core.is_compiled_with_ngraph(
        ) and fluid.core.globals()['FLAGS_use_ngraph']
        if use_ngraph:
            cpu_only = True
        if core.is_compiled_with_cuda() and core.op_support_gpu(self.op_type)\
           and not cpu_only:
            places.append(core.CUDAPlace(0))
        return places

    def check_output(self,
                     atol=1e-5,
                     no_check_set=None,
                     equal_nan=False,
                     check_dygraph=True,
                     inplace_atol=None,
                     check_compile_vs_runtime=True):
        places = self._get_places()
        for place in places:
            res = self.check_output_with_place(place, atol, no_check_set,
                                               equal_nan, check_dygraph)
            if check_dygraph:
                outs, dygraph_outs, fetch_list = res
            else:
                outs, fetch_list = res
            if check_compile_vs_runtime and (
                    self.op_type not in
                    compile_vs_runtime_white_list.COMPILE_RUN_OP_WHITE_LIST):
                self.check_compile_vs_runtime(fetch_list, outs)

    def check_output_customized(self, checker):
        places = self._get_places()
        for place in places:
            outs = self.calc_output(place)
            outs = [np.array(out) for out in outs]
            outs.sort(key=len)
            checker(outs)

    def _assert_is_close(self, numeric_grads, analytic_grads, names,
                         max_relative_error, msg_prefix):

        for a, b, name in six.moves.zip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return ("%s Variable %s max gradient diff %f over limit %f, "
                        "the first error element is %d, expected %f, but got %f"
                        ) % (msg_prefix, name, max_diff, max_relative_error,
                             offset, a.flatten()[offset], b.flatten()[offset])

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   numeric_grad_delta=0.005,
                   in_place=False,
                   max_relative_error=0.005,
                   user_defined_grads=None,
                   check_dygraph=True):
        places = self._get_places()
        for place in places:
            self.check_grad_with_place(place, inputs_to_check, output_names,
                                       no_grad_set, numeric_grad_delta,
                                       in_place, max_relative_error,
                                       user_defined_grads, check_dygraph)

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None,
                              check_dygraph=True):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()

        cache_list = None
        if hasattr(self, "cache_name_list"):
            cache_list = self.cache_name_list
        self.op = create_op(
            self.scope,
            self.op_type,
            op_inputs,
            op_outputs,
            op_attrs,
            cache_list=cache_list)

        if no_grad_set is None:
            no_grad_set = set()

        if not type(output_names) is list:
            output_names = [output_names]

        numeric_grads = user_defined_grads or [
            get_numeric_gradient(
                place,
                self.scope,
                self.op,
                self.inputs,
                input_to_check,
                output_names,
                delta=numeric_grad_delta,
                in_place=in_place) for input_to_check in inputs_to_check
        ]
        analytic_grads = self._get_gradient(inputs_to_check, place,
                                            output_names, no_grad_set)
        self._assert_is_close(numeric_grads, analytic_grads, inputs_to_check,
                              max_relative_error,
                              "Gradient Check On %s" % str(place))

        if check_dygraph:
            dygraph_grad = self._get_dygraph_grad(inputs_to_check, place,
                                                  output_names, no_grad_set)
            self._assert_is_close(numeric_grads, dygraph_grad, inputs_to_check,
                                  max_relative_error,
                                  "Gradient Check On %s" % str(place))

    def _find_var_in_dygraph(self, output_vars, name):
        if name in output_vars:
            return output_vars[name]
        else:
            for output_vars_index in output_vars:
                for output_vars_selected in output_vars[output_vars_index]:
                    if output_vars_selected.name == name:
                        return output_vars_selected

    def _get_dygraph_grad(self,
                          inputs_to_check,
                          place,
                          output_names,
                          no_grad_set=None):
        with fluid.dygraph.base.guard(place=place):
            block = fluid.default_main_program().global_block()

            op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

            # prepare input variable
            inputs, inputs_grad_dict = self.append_input_output_for_dygraph(
                op_proto, self.inputs, True, True, block)

            # prepare output variable
            outputs = self.append_input_output_for_dygraph(
                op_proto, self.outputs, False, False, block)

            # prepare attrbutes
            attrs_outputs = {}
            if hasattr(self, "attrs"):
                for attrs_name in self.attrs:
                    if self.attrs[attrs_name] is not None:
                        attrs_outputs[attrs_name] = self.attrs[attrs_name]
            block.append_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs_outputs if hasattr(self, "attrs") else None)

            outputs_valid = {}
            for output_name in output_names:
                outputs_valid[output_name] = self._find_var_in_dygraph(
                    outputs, output_name)

            if len(outputs_valid) == 1:
                loss = block.create_var(
                    dtype=self.dtype,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
                    stop_gradient=False,
                    shape=[1])
                for outputs_valid_key in outputs_valid:
                    block.append_op(
                        type="mean",
                        inputs={"X": outputs_valid[outputs_valid_key]},
                        outputs={"Out": [loss]},
                        attrs=None)
            else:
                avg_sum = []
                for cur_loss in outputs_valid:
                    cur_avg_loss = block.create_var(
                        dtype=self.dtype,
                        type=core.VarDesc.VarType.LOD_TENSOR,
                        persistable=False,
                        stop_gradient=False)
                    block.append_op(
                        type="mean",
                        inputs={"X": outputs_valid[cur_loss]},
                        outputs={"Out": [cur_avg_loss]},
                        attrs=None)
                    avg_sum.append(cur_avg_loss)
                loss_sum = block.create_var(
                    dtype=self.dtype,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
                    stop_gradient=False,
                    shape=[1])
                block.append_op(
                    type='sum',
                    inputs={"X": avg_sum},
                    outputs={"Out": loss_sum},
                    attrs=None)
                loss = block.create_var(
                    dtype=self.dtype,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
                    stop_gradient=False,
                    shape=[1])
                block.append_op(
                    type='scale',
                    inputs={"X": loss_sum},
                    outputs={"Out": loss},
                    attrs={'scale': 1.0 / float(len(avg_sum))})
            loss.backward()

            fetch_list_grad = []
            for inputs_to_check_name in inputs_to_check:
                a = inputs_grad_dict[inputs_to_check_name].gradient()
                fetch_list_grad.append(a)
            return fetch_list_grad

    @staticmethod
    def _numpy_to_lod_tensor(np_value, lod, place):
        tensor = core.LoDTensor()
        tensor.set(np_value, place)
        if lod is not None:
            tensor.set_recursive_sequence_lengths(lod)
        return tensor

    @staticmethod
    def np_dtype_to_fluid_dtype(input):
        return input

    @staticmethod
    def fluid_dtype_to_np_dtype(self, dtype):
        return dtype

    @staticmethod
    def np_value_to_fluid_value(input):
        return input

    def _get_gradient(self,
                      input_to_check,
                      place,
                      output_names,
                      no_grad_set,
                      parallel=False):
        prog = Program()
        block = prog.global_block()
        self._append_ops(block)
        loss = append_loss_ops(block, output_names)
        param_grad_list = append_backward(
            loss=loss, parameter_list=input_to_check, no_grad_set=no_grad_set)

        inputs = self._get_inputs(block)
        feed_dict = self.feed_var(inputs, place)

        fetch_list = [g for p, g in param_grad_list]
        if parallel:
            use_cuda = False
            if isinstance(place, fluid.CUDAPlace):
                use_cuda = True
            compiled_prog = fluid.CompiledProgram(prog).with_data_parallel(
                loss_name=loss.name, places=place)
            prog = compiled_prog
        executor = fluid.Executor(place)
        return list(
            map(np.array,
                executor.run(prog, feed_dict, fetch_list, return_numpy=False)))


'''
The op test with int8 precision should inherit OpTestInt8.
'''


class OpTestInt8(OpTestBase):
    pass


'''
The op test with float16 precision should inherit OpTestFp16,
which requires the test to call check_grad. 
'''


class OpTestFp16(OpTestBase):
    def check_output(self,
                     atol=1e-5,
                     no_check_set=None,
                     equal_nan=False,
                     check_dygraph=True,
                     inplace_atol=None,
                     check_compile_vs_runtime=False):
        self.__class__.op_type = self.op_type
        OpTestBase.check_output(self, atol, no_check_set, equal_nan,
                                check_dygraph, inplace_atol,
                                check_compile_vs_runtime)

    def _check_grad_helper(self):
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        assert self.dtype == np.float16, "The dtype of this test should be float16."

        self.__class__.op_type = self.op_type
        self.__class__.exist_check_grad = True

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   numeric_grad_delta=0.005,
                   in_place=False,
                   max_relative_error=0.005,
                   user_defined_grads=None,
                   check_dygraph=True):
        self._check_grad_helper()
        OpTestBase.check_grad(self, inputs_to_check, output_names, no_grad_set,
                              numeric_grad_delta, in_place, max_relative_error,
                              user_defined_grads, check_dygraph)

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None,
                              check_dygraph=True):
        self._check_grad_helper()
        OpTestBase.check_grad_with_place(
            self, place, inputs_to_check, output_names, no_grad_set,
            numeric_grad_delta, in_place, max_relative_error,
            user_defined_grads, check_dygraph)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        if not hasattr(cls, "exist_check_grad") \
            and cls.__name__ not in op_check_grad_white_list.NO_NEED_CHECK_GRAD_CASES \
            and cls.op_type not in op_check_grad_white_list.EMPTY_GRAD_OP_LIST:
            raise AssertionError("This test of %s op needs check_grad." %
                                 cls.op_type)


'''
The op test with float32/64 precision should inherit OpTest,
which requires the test to call check_grad with float64 precision. 
'''


class OpTest(OpTestBase):
    def check_output(self,
                     atol=1e-5,
                     no_check_set=None,
                     equal_nan=False,
                     check_dygraph=True,
                     inplace_atol=None,
                     check_compile_vs_runtime=False):
        self.__class__.op_type = self.op_type
        OpTestBase.check_output(self, atol, no_check_set, equal_nan,
                                check_dygraph, inplace_atol,
                                check_compile_vs_runtime)

    def _check_grad_helper(self):
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        assert self.dtype in [np.float16, np.float32, np.float64], \
            "self.dtype = %s." % self.dtype
        if self.dtype == np.float16 and \
            self.op_type not in op_accuracy_white_list.FP16_CHECK_OP_LIST:
            raise AssertionError("The dtype of this test should be float32 "
                                 "or float64. op: %s dtype: %s." %
                                 (self.op_type, self.dtype))

        self.__class__.op_type = self.op_type
        if self.dtype == np.float64:
            self.__class__.exist_fp64_check_grad = True

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   numeric_grad_delta=0.005,
                   in_place=False,
                   max_relative_error=0.005,
                   user_defined_grads=None,
                   check_dygraph=True):
        self._check_grad_helper()
        OpTestBase.check_grad(self, inputs_to_check, output_names, no_grad_set,
                              numeric_grad_delta, in_place, max_relative_error,
                              user_defined_grads, check_dygraph)

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None,
                              check_dygraph=True):
        self._check_grad_helper()
        OpTestBase.check_grad_with_place(
            self, place, inputs_to_check, output_names, no_grad_set,
            numeric_grad_delta, in_place, max_relative_error,
            user_defined_grads, check_dygraph)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        # only for pass ci, but cases in NO_FP64_CHECK_GRAD_CASES 
        # and op in NO_FP64_CHECK_GRAD_OP_LIST should be fixed
        if cls.__name__ not in op_accuracy_white_list.NO_FP64_CHECK_GRAD_CASES \
            and not hasattr(cls, 'exist_fp64_check_grad') \
            and cls.__name__ not in op_check_grad_white_list.NO_NEED_CHECK_GRAD_CASES \
            and cls.op_type not in op_check_grad_white_list.EMPTY_GRAD_OP_LIST \
            and cls.op_type not in op_accuracy_white_list.NO_FP64_CHECK_GRAD_OP_LIST:
            raise AssertionError("This test of %s op needs fp64 check_grad." %
                                 cls.op_type)
