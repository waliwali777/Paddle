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

from collections.abc import Sequence

import paddle
from paddle.distribution import distribution


class Binomial(distribution.Distribution):
    r"""
    The Binomial distribution with size `total_count` and `probability` parameters.

    In probability theory and statistics, the binomial distribution is the most basic discrete probability distribution defined on :math:`[0, n] \cap \mathbb{N}`,
    which can be viewed as the number of times a potentially unfair coin is tossed to get heads, and the result
    of its random variable can be viewed as the sum of a series of independent Bernoulli experiments.

    The probability mass function (pmf) is

    .. math::

        pmf(x; n, p) = \frac{n!}{x!(n-x)!}p^{x}(1-p)^{n-x}

    In the above equation:

    * :math:`total_count = n`: is the size, meaning the total number of Bernoulli experiments.
    * :math:`probability = p`: is the probability of the event happening in one Bernoulli experiments.

    Args:
        total_count(int|Tensor): The size of Binomial distribution which should be greater than 0, meaning the number of independent bernoulli
            trials with probability parameter :math:`p`. The data type will be converted to 1-D Tensor with paddle global default dtype if the input
            :attr:`probability` is not Tensor, otherwise will be converted to the same as :attr:`probability`.
        probability(float|Tensor): The probability of Binomial distribution which should reside in [0, 1], meaning the probability of success
            for each individual bernoulli trial. If the input data type is float, it will be converted to a 1-D Tensor with paddle global default dtype.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Binomial
            >>> rv = Binomial(100, paddle.to_tensor([0.3, 0.6, 0.9]))

            >>> # doctest: +SKIP
            >>> print(rv.sample([2]))
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[33., 56., 93.],
            [32., 53., 91.]])

            >>> # doctest: -SKIP
            >>> print(rv.mean)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [30.00000191, 60.00000381, 90.        ])

            >>> print(rv.entropy())
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.94057941, 3.00785327, 2.51125669])
    """

    def __init__(self, total_count, probability):
        self.dtype = paddle.get_default_dtype()
        self.total_count, self.probability = self._to_tensor(
            total_count, probability
        )

        if not self._check_constraint(self.total_count, self.probability):
            raise ValueError(
                'Every element of input parameter `total_count` should be grater than or equal to one, and `probability` should be grater than or equal to zero and less than or equal to one.'
            )
        if self.total_count.shape == []:
            batch_shape = (1,)
        else:
            batch_shape = self.total_count.shape
        super().__init__(batch_shape)

    def _to_tensor(self, total_count, probability):
        """Convert the input parameters into Tensors if they were not and broadcast them

        Returns:
            Tuple[Tensor, Tensor]: converted total_count and probability.
        """
        # convert type
        if isinstance(probability, float):
            probability = paddle.to_tensor(probability, dtype=self.dtype)
        else:
            self.dtype = probability.dtype
        if isinstance(total_count, int):
            total_count = paddle.to_tensor(total_count, dtype=self.dtype)
        else:
            total_count = paddle.cast(total_count, dtype=self.dtype)

        # broadcast tensor
        return paddle.broadcast_tensors([total_count, probability])

    def _check_constraint(self, total_count, probability):
        """Check the constraints for input parameters

        Args:
            total_count (Tensor)
            probability (Tensor)

        Returns:
            bool: pass or not.
        """
        total_count_check = (total_count >= 1).all()
        probability_check = (probability >= 0).all() * (probability <= 1).all()
        return total_count_check and probability_check

    @property
    def mean(self):
        """Mean of binomial distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.total_count * self.probability

    @property
    def variance(self):
        """Variance of binomial distribution.

        Returns:
            Tensor: variance value.
        """
        return self.total_count * self.probability * (1 - self.probability)

    def sample(self, shape=()):
        """Generate binomial samples of the specified shape. The final shape would be ``shape+batch_shape`` .

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape`. The returned data type is int64.
        """
        if not isinstance(shape, Sequence):
            raise TypeError('sample shape must be Sequence object.')

        with paddle.set_grad_enabled(False):
            shape = tuple(shape)
            batch_shape = tuple(self.batch_shape)
            output_shape = tuple(shape + batch_shape)
            output_size = paddle.broadcast_to(
                self.total_count, shape=output_shape
            )
            output_prob = paddle.broadcast_to(
                self.probability, shape=output_shape
            )
            return paddle.binomial(output_size, output_prob)

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            \mathcal{H}(X) = - \sum_{x \in \Omega} p(x) \log{p(x)}

        In the above equation:

        * :math:`\Omega`: is the support of the distribution.

        Args:
            n (float): size of the binomial r.v.
            p (float): probability of the binomial r.v.

        Returns:
            Tensor: Shannon entropy of binomial distribution. The data type is the same as `probability`.
        """
        values = self._enumerate_support()
        log_prob = self.log_prob(values)
        return -(paddle.exp(log_prob) * log_prob).sum(0)

    def _enumerate_support(self):
        """Return the support of binomial distribution [0, 1, ... ,n]

        Returns:
            Tensor: the support of binomial distribution
        """
        values = paddle.arange(
            1 + paddle.max(self.total_count), dtype=self.dtype
        )
        values = values.reshape((-1,) + (1,) * len(self.batch_shape))
        return values

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is the same as `probability`.
        """
        value = paddle.cast(value, dtype=self.dtype)

        # combination
        log_comb = (
            paddle.lgamma(self.total_count + 1.0)
            - paddle.lgamma(self.total_count - value + 1.0)
            - paddle.lgamma(value + 1.0)
        )
        eps = paddle.finfo(self.probability.dtype).eps
        probs = paddle.clip(self.probability, min=eps, max=1 - eps)
        # log_p
        return paddle.nan_to_num(
            (
                log_comb
                + value * paddle.log(probs)
                + (self.total_count - value) * paddle.log(1 - probs)
            ),
            neginf=-eps,
        )

    def prob(self, value):
        """Probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability. The data type is the same as `probability`.
        """
        return paddle.exp(self.log_prob(value))

    def kl_divergence(self, other):
        r"""The KL-divergence between two binomial distributions with the same :attr:`total_count`.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(n_1, p_1, n_2, p_2) = \sum_x p_1(x) \log{\frac{p_1(x)}{p_2(x)}}

        .. math::

            p_1(x) = \frac{n_1!}{x!(n_1-x)!}p_1^{x}(1-p_1)^{n_1-x}

        .. math::

            p_2(x) = \frac{n_2!}{x!(n_2-x)!}p_2^{x}(1-p_2)^{n_2-x}

        Args:
            other (Binomial): instance of ``Binomial``.

        Returns:
            Tensor: kl-divergence between two binomial distributions. The data type is the same as `probability`.

        """
        if not (paddle.equal(self.total_count, other.total_count)).all():
            raise ValueError(
                "KL divergence of two binomial distributions should share the same `total_count` and `batch_shape`."
            )
        support = self._enumerate_support()
        log_prob_1 = self.log_prob(support)
        log_prob_2 = other.log_prob(support)
        return (
            paddle.multiply(
                paddle.exp(log_prob_1),
                (paddle.subtract(log_prob_1, log_prob_2)),
            )
        ).sum(0)
