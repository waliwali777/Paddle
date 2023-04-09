#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from . import decorator
from .decorator import decorate, amp_decorate
from . import fp16_lists
from .fp16_lists import *
from . import fp16_utils
from .fp16_utils import *
from . import bf16

__all__ = []
__all__ += decorator.__all__
__all__ += fp16_lists.__all__
__all__ += fp16_utils.__all__


_global_amp_dtype = None


def set_global_amp_dtype(dtype):
    # check amp_dtype: float16 or bfloat16
    dtype = dtype.lower()
    if not (dtype in ['float16', 'bfloat16']):
        raise ValueError(
            "If enable amp, dtype should be 'float16' or 'bfloat16'."
        )

    global _global_amp_dtype
    warnings.warn(
        f"Set global amp_dtype to {dtype}, the previous value is {_global_amp_dtype}"
    )
    _global_amp_dtype = dtype


def get_global_amp_dtype():
    global _global_amp_dtype
    return _global_amp_dtype
