# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid import unique_name, core
from .primops import fill_const, add
from .primrules import get_input_vars, get_output_vars, _orig2prim, _prim2orig, _jvp, _transpose
from collections import OrderedDict


def topo_path(xs, ys, block=None):
    """ Returns the ops in topological on the paths from input `xs` to 
    output `ys`. """

    if block is None:
        block = default_main_program().current_block()
    path = []
    def_vars = list(xs)
    sink_ops = {}
    for op in block.ops:
        if len(sink_ops) == len(ys):
            break
        ins = set(get_input_vars(op))
        if any(ins.intersection(def_vars)):
            path.append(op)
            outs = set(get_output_vars(op))
            for out in outs:
                if any(out is y for y in ys):
                    # Found an output op
                    assert not any(out is y for y in sink_ops)
                    sink_ops[id(out)] = op
                else:
                    def_vars.append(out)
    if len(sink_ops) != len(ys):
        not_reachable = (var for var in ys if id(var) not in sink_ops)
        raise f"Output vars: {' '.join(not_reachable)} are not reachable from inputs."
    return path


def vars_on_path(xs, ys, block=None):
    
    if block is None:
        block = default_main_program().current_block()

    vars = OrderedDict()
    
    for var in xs + ys:
        vars[id(var)] = var

    sink_ops = set(y.op for y in ys)

    for op in topo_path(xs, ys, block):
        if op not in sink_ops:
            for out in get_output_vars(op):
                vars[id(out)] = out

    return vars
class VarMap(object):
    __slots__ = ['name', 'varset', 'tab']

    def __init__(self, name, varset):
        self.name = name
        self.varset = varset
        self.tab = OrderedDict()
    
    def add(self, key_var, value_var):
        self.tab[id(key_var)] = id(value_var)
  
    def lookup(self, key_var):
        value_id = self.tab.get(id(key_var))
        if value_id is not None:
            return self.varset.get(value_id)
        else:
            return None

    def delete(self, key_var):
        del self.tab[id(key_var)]

    def delete_keyvars(self, key_vars):
        for var in key_vars:
            del self.tab[id(var)]

    def delete_valuevars(self, value_vars):
        ids = [id(v) for v in value_vars]
        keys = [k for k, v in self.tab.items() if v in ids]
        for k in keys:
            del self.tab[k]

    def contain_var(self, key_var):
        return self.tab.__contains__(id(key_var))

    def contain_value(self, value_var):
        return id(value_var) in self.tab.values()

class Transform(object):
    """ An object that maintains the state of transformations applied to a 
    primitve program. """

    def __init__(self, block):
        self.block = block
        self.vars = self.init_vars(block)
        self.var2dot = VarMap('var2dot', self.vars)
        self.dot2bar = VarMap('dot2var', self.vars)

    def init_vars(self, block):
        vars = OrderedDict()
        for _, var in block.vars.items():
            vars[id(var)] = var
        return vars

    def add_vars(self, new_vars):
        self.vars.update({id(v) : v for v in new_vars if v is not None})

    def erase_dots(self, vars_to_erase):
        for var in vars_to_erase:
            if id(var) in self.vars:
                del self.vars[id(var)]
        self.dot2bar.delete_keyvars(vars_to_erase)
        self.var2dot.delete_valuevars(vars_to_erase)
        for var in vars_to_erase:
            del var.block.vars[var.name]

    # def is_dot(self, var):
    #     return self.var2dot.contain_value(var)

    def lower(self, block=None, reverse=False):
        lower_fn = _prim2orig if reverse else _orig2prim
        if block is None:
            program = default_main_program()
            assert program.num_blocks == 1, "The lower transform is designed to process only one block."
            block = program.current_block()

        vlt = {}
        to_bind = {}
        for var in block.desc.all_vars():
            vlt[var.name()] = block.var(var.name())

        ops_to_remove = []
        vars_to_remove = []
        for op_idx in range(len(block.ops)):
            op = block.ops(op_idx)
            if lookup_orig2prim(op.type()) is not None:
                ops_to_remove.append(op_idx)
                input_args = list(get_input_vars(op))
                for i in range(len(input_args)):
                    if input_args[i] is not None and input_args[i].name(
                    ) in to_bind:
                        input_args[i] = to_bind[input_args[i].name()]
                for orig_out, new_out in zip(
                        get_output_vars(op), lower_fn(op, input_args)):
                    assert not (orig_out is None ^ new_out is None
                                ), "orig_out and new_out should match."
                    vars_to_remove.append(orig_out.name())
                    vlt[new_out.name()] = new_out
                    to_bind[orig_out.name()] = new_out.name()

        for op_idx in reversed(ops_to_remove):
            block._remove_op(op_idx)
        for var_name in vars_to_remove:
            block._remove_var(var_name)

    def linearize(self, xs, ys, xs_dot=None):
        if xs_dot is None:
            xs_dot = [fill_const(1.0, shape=x.shape, dtype=x.dtype) for x in xs]
            self.add_vars(xs_dot)
        else:
            assert len(xs) == len(xs_dot)

        for x, dot in zip(xs, xs_dot):
            assert x.dtype == dot.dtype
            assert x.shape == dot.shape
            self.var2dot.add(x, dot)

        for op in topo_path(xs, ys, self.block):
            ins = get_input_vars(op)
            # an input var may not be on the input-output path, therefore
            # there's no forward gradient linked to it. This implies 
            # there may be None's in `ins_dot`. In this case we place
            # the original input in the position of the otherwise forward
            # gradient.
            jvp_ins = []
            for var in ins:
                dot = self.var2dot.lookup(var)
                if dot is None:
                    jvp_ins.append(var)
                else:
                    jvp_ins.append(dot)

            # apply op's forward ad rule
            outs_dot = _jvp(op, *jvp_ins)

            if not isinstance(outs_dot, list):
                outs_dot = [outs_dot]

            self.add_vars(outs_dot)

            outs = get_output_vars(op)
            assert len(outs) == len(outs_dot)
            for out, dot in zip(outs, outs_dot):
                self.var2dot.add(out, dot)
        
        ys_dot = [self.var2dot.lookup(y) for y in ys]
        return xs_dot, ys_dot

    def transpose(self, ys_dot, xs_dot, ys_bar=None, retain_fwd=False):
        if ys_bar is None:
            ys_bar = []
            for y in ys_dot:
                ys_bar.append(fill_const(1.0, shape=y.shape, dtype=y.dtype))
            self.add_vars(ys_bar)
        else:
            assert len(ys_dot) == len(ys_bar)
            for y_dot, y_bar in zip(ys_dot, ys_bar):
                assert y_dot.shape == y_bar.shape
                assert y_dot.dtype == y_bar.dtype

        for dot, bar in zip(ys_dot, ys_bar):
            self.dot2bar.add(dot, bar)
        
        # find all the relevant forward gradients
        dotvars = vars_on_path(xs_dot, ys_dot)

        is_dot = lambda v: id(v) in dotvars

        for op in reversed(topo_path(xs_dot, ys_dot, self.block)):
            outs_bar = [self.dot2bar.lookup(var) for var in get_output_vars(op)]
            ins_bar = _transpose(op, is_dot, *outs_bar)
            if isinstance(ins_bar, (list, tuple)):
                ins_bar = list(ins_bar)
            else:
                ins_bar = [ins_bar]
            self.add_vars(ins_bar)
            for dot, bar in zip(get_input_vars(op), ins_bar):
                if bar is not None:
                    # aggregate gradient
                    grad = self.dot2bar.lookup(dot)
                    if grad is None:
                        self.dot2bar.add(dot, bar)
                    else: 
                        grad = add(grad, bar)
                        self.add_vars([grad])
                        self.dot2bar.add(dot, grad)
                    
        xs_bar = [self.dot2bar.lookup(x) for x in xs_dot]

        if not retain_fwd:
            dots_to_remove = set()
            for op in topo_path(xs_dot, ys_dot):
                for var in get_input_vars(op):
                    if is_dot(var):
                        dots_to_remove.add(var)
                block = op.block
                op_idx = block.ops.index(op)
                block._remove_op(op_idx)

            self.erase_dots(dots_to_remove)

        return ys_bar, xs_bar
