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

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import paddle
from paddle.distribution import constraint

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.distribution.constraint import Constraint


class Variable:
    """Random variable of probability distribution.

    Args:
        is_discrete (bool): Is the variable discrete or continuous.
        event_rank (int): The rank of event dimensions.
    """

    def __init__(
        self,
        is_discrete: bool = False,
        event_rank: int = 0,
        constraint: Constraint | None = None,
    ) -> None:
        self._is_discrete = is_discrete
        self._event_rank = event_rank
        self._constraint = constraint

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    @property
    def event_rank(self) -> int:
        return self._event_rank

    def constraint(self, value: Tensor) -> Tensor:
        """Check whether the 'value' meet the constraint conditions of this
        random variable."""
        assert self._constraint is not None
        return self._constraint(value)


class Real(Variable):
    def __init__(self, event_rank: int = 0) -> None:
        super().__init__(False, event_rank, constraint.real)


class Positive(Variable):
    def __init__(self, event_rank: int = 0) -> None:
        super().__init__(False, event_rank, constraint.positive)


class Independent(Variable):
    """Reinterprets some of the batch axes of variable as event axes.

    Args:
        base (Variable): Base variable.
        reinterpreted_batch_rank (int): The rightmost batch rank to be
            reinterpreted.
    """

    def __init__(self, base: Variable, reinterpreted_batch_rank: int) -> None:
        self._base = base
        self._reinterpreted_batch_rank = reinterpreted_batch_rank
        super().__init__(
            base.is_discrete, base.event_rank + reinterpreted_batch_rank
        )

    def constraint(self, value: Tensor) -> Tensor:
        ret = self._base.constraint(value)
        if ret.dim() < self._reinterpreted_batch_rank:
            raise ValueError(
                f"Input dimensions must be equal or grater than  {self._reinterpreted_batch_rank}"
            )
        return ret.reshape(
            ret.shape[: ret.dim() - self.reinterpreted_batch_rank] + (-1,)
        ).all(-1)


class Stack(Variable):
    def __init__(self, vars: Sequence[Variable], axis: int = 0) -> None:
        self._vars = vars
        self._axis = axis

    @property
    def is_discrete(self) -> bool:
        return any(var.is_discrete for var in self._vars)

    @property
    def event_rank(self) -> int:
        rank = max(var.event_rank for var in self._vars)
        if self._axis + rank < 0:
            rank += 1
        return rank

    def constraint(self, value: Tensor) -> Tensor:
        if not (-value.dim() <= self._axis < value.dim()):
            raise ValueError(
                f'Input dimensions {value.dim()} should be grater than stack '
                f'constraint axis {self._axis}.'
            )

        return paddle.stack(
            [
                var.check(value)
                for var, value in zip(
                    self._vars, paddle.unstack(value, self._axis)
                )
            ],
            self._axis,
        )


real = Real()
positive = Positive()
