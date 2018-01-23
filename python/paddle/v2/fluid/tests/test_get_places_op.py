#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2.fluid as fluid
import decorators
import unittest


class TestGetPlaces(unittest.TestCase):
    @decorators.prog_scope()
    def test_get_places(self):
        places = fluid.layers.get_places()
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_main_program())
        self.assertEqual(places.type, fluid.core.VarDesc.VarType.PLACE_LIST)


if __name__ == '__main__':
    unittest.main()
