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

import test_collective_api_base as test_base

import paddle.distributed as dist
from paddle import fluid


class TestCollectiveBroadcastObjectListAPI(
    test_base.TestCollectiveAPIRunnerBase
):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        with fluid.program_guard(main_prog, startup_program):
            object_list = [indata]
            dist.broadcast_object_list(object_list, src=1)
            return object_list


if __name__ == "__main__":
    test_base.runtime_main(
        TestCollectiveBroadcastObjectListAPI, "broadcast_object_list"
    )
