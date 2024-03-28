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

from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_sharding_parameters,
)
from ..utils.log_util import logger
from .meta_parallel_base import MetaParallelBase

__all__ = []


class ShardingParallel(MetaParallelBase):
    def __init__(self, layers, hcg, **kwargs):
        super().__init__(layers, hcg, **kwargs)

    def _prepare_for_model(self):
        logger.info("start broadcast sharding parameters")
        broadcast_sharding_parameters(self._layers, self._hcg)
        logger.info("sharding's parameters is ready")

        if self._hcg.get_data_parallel_world_size() > 1:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)
            logger.info("dp's parameters is ready")

        # TODO (JZ-LIANG) to support Sharding-DP
