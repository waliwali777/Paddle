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

# from ...fluid.contrib.slim.quantization.imperative.ptq_config import PTQConfig, default_ptq_config
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import BaseQuantizer
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import AbsmaxQuantizer
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import PerChannelAbsmaxQuantizer
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import KLQuantizer
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import HistQuantizer
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import SUPPORT_ACT_QUANTIZERS
# from ...fluid.contrib.slim.quantization.imperative.ptq_quantizer import SUPPORT_WT_QUANTIZERS
# from ...fluid.contrib.slim.quantization.imperative.ptq_registry import PTQRegistry
# from ...fluid.contrib.slim.quantization.imperative.ptq import ImperativePTQ
# from ...fluid.contrib.slim.quantization.imperative.qat import ImperativeQuantAware
from .qat import QAT
from . import qat
from .ptq import PTQ
from . import ptq
from .config import QuantConfig, TRTQuantConfig
from . import config
from .quanters import (
    ActLSQPlusQuanter,
    ActLSQPlusQuanter,
    FakeQuanterWithAbsMaxObserver,
)
from . import quanters
from .stubs import Stub
from . import stubs
from .factory import ObserverFactory, QuanterFactory
from . import factory
from .quanter import BaseQuanter
from . import quanter
from .observer import BaseObserver
from . import observer

__all__ = []
__all__ += qat.__all__
__all__ += ptq.__all__
__all__ += config.__all__
__all__ += quanters.__all__
__all__ += stubs.__all__
__all__ += factory.__all__
__all__ += quanter.__all__
__all__ += observer.__all__
