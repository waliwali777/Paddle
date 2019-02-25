# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import core
import six
import threading
from .framework import Program, Variable, program_guard, default_main_program, default_startup_program
from .executor import global_scope
from .data_feeder import DataFeeder
from .layers.io import monkey_patch_reader_methods, _copy_reader_var_, double_buffer
import unique_name

__all__ = ['PyReader']


def _convert_places(places):
    if not isinstance(places, (list, tuple)):
        places = [places]

    ret = []
    for p in places:
        if not isinstance(p, core.Place):
            tmp = core.Place()
            tmp.set_place(p)
            p = tmp

        ret.append(p)
    return ret


class PyReader(object):
    unique_name_generator = unique_name.UniqueNameGenerator()

    def __init__(self,
                 feed_list,
                 capacity,
                 use_double_buffer=True,
                 iterable=True):
        self._tensor_reader = None
        self._thread = None
        self._iterable = iterable
        self._use_double_buffer = use_double_buffer
        self._capacity = capacity
        self._feed_list = feed_list
        self._scope = global_scope()
        if not self._iterable:
            self._init_non_iterable()

    def _init_iterable(self, places):
        self._var_names = [v.name for v in self._feed_list]
        self._places = _convert_places(places)
        self._queue = core.init_lod_tensor_blocking_queue(core.Variable(),
                                                          self._capacity)
        self._reader = core.create_py_reader(
            self.queue, self._var_names, self._places, self._use_double_buffer)

    def _init_non_iterable(self):
        lod_levels = []
        dtypes = []
        shape_concat = []
        ranks = []
        shapes = []

        for feed_data in self._feed_list:
            dtypes.append(feed_data.dtype)
            shape_concat.extend(feed_data.shape)
            ranks.append(len(feed_data.shape))
            shapes.append(feed_data.shape)
            lod_levels.append(feed_data.lod_level)

        queue_name = PyReader.unique_name_generator('lod_tensor_blocking_queue')
        reader_name = PyReader.unique_name_generator('create_py_reader')
        double_buffer_name = PyReader.unique_name_generator('double_buffer')

        var = self._scope.var(queue_name)
        self._queue = core.init_lod_tensor_blocking_queue(var, self._capacity)

        startup_blk = default_startup_program().current_block()
        startup_var = startup_blk.create_var(name=reader_name)

        startup_blk.append_op(
            type='create_py_reader',
            inputs={'blocking_queue': [queue_name]},
            outputs={'Out': [startup_var]},
            attrs={
                'shape_concat': shape_concat,
                'lod_levels': lod_levels,
                'ranks': ranks
            })

        startup_var.desc.set_dtypes(dtypes)
        startup_var.persistable = True

        main_prog_var = _copy_reader_var_(
            default_main_program().current_block(), startup_var)

        main_prog_var.stop_gradient = True
        main_prog_var.persistable = True

        reader = monkey_patch_reader_methods(main_prog_var)
        if self._use_double_buffer:
            double_buffer_reader = double_buffer(
                reader, name=double_buffer_name)
            # we return a double buffer reader. However, the reset method comes from
            # py_reader.
            double_buffer_reader.reset = reader.reset
            reader = double_buffer_reader

        self._reader = reader

        default_main_program().current_block().append_op(
            type='read',
            inputs={'Reader': [self._reader]},
            outputs={'Out': self._feed_list})

    @property
    def queue(self):
        return self._queue

    @property
    def iterable(self):
        return self._iterable

    def __call__(self):
        assert self.iterable, "PyReader is not iterable"
        assert self._tensor_reader is not None, \
            "Data source of PyReader has not set yet"

        class Iterator(object):
            def __init__(self, reader):
                self._reader = reader._reader
                self._reset = reader._reset

            def __iter__(self):
                return self

            def next(self):
                ret = self._reader.read_next()
                if ret:
                    return ret
                else:
                    self._reset()
                    raise StopIteration

        self._start()
        return Iterator(self)

    def _reset(self):
        self._reader.reset()
        self._thread.join()

    def start(self):
        assert not self._iterable, "start() cannot be called when PyReader is iterable"
        self._start()

    def reset(self):
        assert not self._iterable, "reset() cannot be called when PyReader is iterable"
        self._reset()

    def _start(self):
        def __thread_main__():
            for tensors in self._tensor_reader():
                array = core.LoDTensorArray()
                for item in tensors:
                    if not isinstance(item, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(item, core.CPUPlace())
                        item = tmp

                    array.append(item)

                if not self._queue.push(array):
                    break

            self._queue.close()

        self._thread = threading.Thread(target=__thread_main__)
        self._thread.daemon = True
        self._thread.start()

    def decorate_paddle_reader(self, reader, places=None):
        assert self._tensor_reader is None, \
            "Cannot reset the data source of PyReader"
        with program_guard(Program(), Program()):
            feeder = DataFeeder(
                feed_list=self._feed_list, place=core.CPUPlace())
            paddle_reader = feeder.decorate_reader(reader, multi_devices=False)

        def __tensor_reader_impl__():
            for slots in paddle_reader():
                yield [slots[var.name] for var in self._feed_list]

        self.decorate_tensor_provider(__tensor_reader_impl__, places)

    def decorate_tensor_provider(self, reader, places=None):
        assert self._tensor_reader is None, \
            "Cannot reset the data source of PyReader"
        self._tensor_reader = reader
        if self._iterable:
            assert places is not None, "Places cannot be None when py_reader is iterable"
            self._init_iterable(places)
