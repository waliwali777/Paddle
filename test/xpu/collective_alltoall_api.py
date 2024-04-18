# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import test_collective_api_base as test_base

import paddle
from paddle import base

paddle.enable_static()


class TestCollectiveAllToAllAPI(test_base.TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype='float32'):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype=dtype
            )
            tindata.desc.set_need_check_feed(False)
            tindata = paddle.split(tindata, 2, axis=0)
            tout_data = []
            paddle.distributed.alltoall(tindata, tout_data)
            return tout_data


if __name__ == "__main__":
    test_base.runtime_main(TestCollectiveAllToAllAPI, "alltoall")
