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

__all__ = ['convert_call']

import collections
import copy
import functools
import inspect
import pdb
import re
import types

import numpy
import six

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.dygraph_to_static.program_translator import PartialProgram
from paddle.fluid.dygraph.dygraph_to_static.program_translator import convert_to_static
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.dygraph.dygraph_to_static.convert_operators import convert_len

DECORATOR_NAMES = ['declarative', 'dygraph_to_static_func']
program_translator = ProgramTranslator()


def is_builtin(func):
    if isinstance(func, types.BuiltinFunctionType):
        return True
    elif func in six.moves.builtins.__dict__.values():
        return True
    # Other built-in modules
    # TODO(liym27): A better way to do this.
    elif any(func in m.__dict__.values()
             for m in (collections, pdb, copy, inspect, re, six, numpy)):
        return True
    else:
        return False


def is_builtin_len(func):
    if isinstance(func, types.BuiltinFunctionType) and func.__name__ == 'len':
        return True
    return False


def is_paddle_func(func):
    m = inspect.getmodule(func)
    return m is not None and m.__name__.startswith("paddle")


def convert_call(func):
    """
    Converts a function call which needs to be transformed to static fucntion.

    Args:
        func (callable): A callable function or method to convert.

    Returns:
        Callable: A converted function.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.dygraph_to_static import convert_call

          def dyfunc(x):
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1

               return x_v
          new_func = convert_call(dyfunc)
          x = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='float64')
          x_v = new_func(x)
          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fetch_list=[x_v])
          print(out[0])
          # [[1. 1. 1.]
          #  [1. 1. 1.]
          #  [1. 1. 1.]]

    """
    func_self = None
    converted_call = None

    # Function in convert_call may be decorated by another `@declarative`,
    # in this case, unwraps it into a raw method or function.
    if isinstance(func, PartialProgram):
        instance = func._class_instance
        if instance is not None:
            func = func._dygraph_func.__get__(instance)
        else:
            func = func._dygraph_func

    if is_builtin_len(func):
        return convert_len

    if is_builtin(func) or is_paddle_func(func):
        return func

    if inspect.isfunction(func):
        # TODO(liym27): If func is a lambda function, special conversion is needed.
        if func.__name__ == '<lambda>':
            return func
        try:
            # Note(Aurelius84): Because `@declarative` return a class instance instead of
            # a function. This will modify the value refer to itself in `__globals__`.

            # For example: 
            #
            #      @declarative
            #      def foo(x):
            #          return x
            #
            # `foo` will be converted into a wrapper class, suppose as `PartialProgram`.
            # And `foo.__globals__['foo']` will still return this `PartialProgram` instead of
            # `foo` function.
            # So `isinstance(fn, PartialProgram)` is added here. 
            global_funcs = set()
            for fn in func.__globals__.values():
                if inspect.isfunction(fn):
                    global_funcs.add(fn)
                elif isinstance(fn, PartialProgram):
                    global_funcs.add(fn._dygraph_func)

            if func in global_funcs:
                converted_call = convert_to_static(func)
                func_self = getattr(func, '__self__', None)
        except AttributeError:
            # NOTE:
            # If func is not in __globals__, it does not need to be transformed
            # because it has been transformed before.
            converted_call = None
        except (IOError, OSError):
            # NOTE:
            # If func has been decorated, its source code can not be get
            # so that it can not be transformed to static function.
            converted_call = None
    elif inspect.ismethod(func):
        try:
            converted_call = convert_to_static(func)
            func_self = getattr(func, '__self__', None)
        except (IOError, OSError):
            # NOTE: func may have been decorated.
            converted_call = None

    elif hasattr(func, '__class__') and hasattr(func.__class__, '__call__'):
        if hasattr(func, 'forward') and isinstance(func, Layer):
            try:
                forward_func = convert_to_static(func.forward)
                setattr(func, 'forward', forward_func)
                func_self = func
            except Exception:
                # NOTE: func.forward may have been decorated.
                func_self = None if func_self else func_self
            converted_call = func
        else:
            try:
                call_func = func.__class__.__call__
                converted_call = convert_to_static(call_func)
                func_self = func
            except Exception:
                # NOTE:
                # If `func` is a class which is being initialized, for example `convert_call(Foo)()`,
                # it doesn't need to be transformed
                func_self = None if func_self else func_self

    if converted_call is None:
        return func

    if func_self:
        converted_call = functools.partial(converted_call, func_self)
    return converted_call
