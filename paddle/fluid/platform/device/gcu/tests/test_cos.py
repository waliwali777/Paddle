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

import numpy as np
import pytest
from api_base import ApiBase

import paddle

test = ApiBase(func=paddle.cos, feed_names=['data'], feed_shapes=[[4]])


@pytest.mark.cos
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_cos():
    data = np.array([-0.4, -0.2, 0.1, 0.3], dtype=np.float32)
    test.run(feed=[data])
