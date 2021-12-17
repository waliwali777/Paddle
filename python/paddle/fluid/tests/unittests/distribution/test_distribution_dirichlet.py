# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle
import scipy.stats

from config import (ATOL, DEVICES, RTOL, TEST_CASE_NAME, parameterize, place,
                    xrand)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'concentration'),
              [('test-one-dim', np.random.rand(89))])
class TestDirichlet(unittest.TestCase):
    def setUp(self):
        self._paddle_diric = paddle.distribution.Dirichlet(
            paddle.to_tensor(self.concentration))

    def test_mean(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_diric.mean,
                scipy.stats.dirichlet.mean(self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))

    def test_variance(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_diric.variance,
                scipy.stats.dirichlet.var(self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))

    def test_prob(self):
        value = [np.random.rand(*self.concentration.shape)]
        value = [v / v.sum() for v in value]

        for v in value:
            with paddle.fluid.dygraph.guard(self.place):
                np.testing.assert_allclose(
                    self._paddle_diric.prob(paddle.to_tensor(v)),
                    scipy.stats.dirichlet.pdf(v, self.concentration),
                    rtol=RTOL.get(str(self.concentration.dtype)),
                    atol=ATOL.get(str(self.concentration.dtype)))

    def test_log_prob(self):
        value = [np.random.rand(*self.concentration.shape)]
        value = [v / v.sum() for v in value]

        for v in value:
            with paddle.fluid.dygraph.guard(self.place):
                np.testing.assert_allclose(
                    self._paddle_diric.log_prob(paddle.to_tensor(v)),
                    scipy.stats.dirichlet.logpdf(v, self.concentration),
                    rtol=RTOL.get(str(self.concentration.dtype)),
                    atol=ATOL.get(str(self.concentration.dtype)))

    def test_entropy(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_diric.entropy(),
                scipy.stats.dirichlet.entropy(self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))


if __name__ == '__main__':
    unittest.main()
