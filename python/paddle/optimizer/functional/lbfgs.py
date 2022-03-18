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

import numpy as np

from line_search import strong_wolfe
from utils import _value_and_gradient

import paddle
from paddle.fluid.framework import in_dygraph_mode


def miminize_lbfgs(objective_func,
                   initial_position,
                   history_size=100,
                   max_iters=50,
                   tolerance_grad=1e-8,
                   tolerance_change=1e-8,
                   initial_inverse_hessian_estimate=None,
                   line_search_fn='strong_wolfe',
                   max_line_search_iters=50,
                   initial_step_length=1.0,
                   dtype='float32',
                   name=None):
    r"""Minimizes a differentiable function `func` using the L-BFGS method.
    The L-BFGS is simalar as BFGS, the only difference is that L-BFGS use historical
    sk, yk, rhok rather than H_k-1 to compute Hk.
    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization,
        Second Edition, 2006.
    Args:
        objective_func: the objective function to minimize. ``func`` accepts
            a multivariate input and returns a scalar.
        initial_position (Tensor): the starting point of the iterates. For methods like Newton and quasi-Newton 
        the initial trial step length should always be 1.0 .
        history_size (Scalar): the number of stored vector pairs {si,yi}.
        max_iters (Scalar): the maximum number of minimization iterations.
        tolerance_grad (Scalar): terminates if the gradient norm is smaller than
            this. Currently gradient norm uses inf norm.
        tolerance_change (Scalar): terminates if the change of function value/position/parameter between 
            two iterations is smaller than this value.
        initial_inverse_hessian_estimate (Tensor): the initial inverse hessian approximation.
        line_search_fn (str): indicate which line search method to use, 'strong wolfe' or 'hager zhang'. 
            only support 'strong wolfe' right now.
        max_line_search_iters (Scalar): the maximum number of line search iterations.
        initial_step_length: step length used in first iteration of line search. different initial_step_length 
        may cause different optimal result.
        dtype ('float' | 'float32' | 'float64' | 'double'): the data
            type to be used.
    
    Returns:
        is_converge (bool): Indicates whether found the minimum within tolerance.
        num_func_calls (int): number of objective function called.
        position (Tensor): the position of the last iteration. If the search converged, this value is the argmin of 
        the objective function regrading to the initial position.
        objective_value (Tensor): objective function value at the `position`.
        objective_gradient (Tensor): objective function gradient at the `position`.
    """

    I = paddle.eye(initial_position.shape[0], dtype=dtype)
    if initial_inverse_hessian_estimate is None:
        H0 = I
    else:
        H0 = paddle.assign(initial_inverse_hessian_estimate)
        is_symmetric = paddle.all(paddle.equal(H0, H0.t()))
        # In static mode, raise is not supported, but cholesky will throw preconditionNotMet if 
        # H0 is not symmetric or positive definite.
        if not is_symmetric:
            raise ValueError(
                "The initial_inverse_hessian_estimate should be symmetric, but the specified is not.\n{}".
                format(H0))
        try:
            paddle.linalg.cholesky(H0)
        except RuntimeError as error:
            raise ValueError(
                "The initial_inverse_hessian_estimate should be positive definite, but the specified is not.\n{}".
                format(H0))

    xk = paddle.assign(initial_position)
    value, g1 = _value_and_gradient(objective_func, xk)
    num_func_calls = paddle.assign(1)
    if in_dygraph_mode():
        is_converge = False
        k = 0
        sk_vec = []
        yk_vec = []
        rhok_vec = []
        ai_vec = [None] * history_size
        while k < max_iters:
            vec_len = len(sk_vec)
            q = g1
            for i in range(vec_len - 1, -1, -1):
                ai_vec[i] = rhok_vec[i] * paddle.dot(sk_vec[i], q)
                q = q - ai_vec[i] * yk_vec[i]

            r = paddle.matmul(H0, q)

            for i in range(vec_len):
                beta = rhok_vec[i] * paddle.dot(yk_vec[i], r)
                r = r + sk_vec[i] * (ai_vec[i] - beta)

            pk = -r

            if line_search_fn == 'strong_wolfe':
                alpha, value, g2, ls_func_calls = strong_wolfe(
                    f=objective_func,
                    xk=xk,
                    pk=pk,
                    initial_step_length=initial_step_length)
            else:
                raise NotImplementedError(
                    "Currently only support line_search_fn = 'strong_wolfe', but the specified is '{}'".
                    format(line_search_fn))
            num_func_calls += ls_func_calls

            sk = alpha * pk
            yk = g2 - g1

            rhok = 1. / paddle.dot(yk, sk)
            if paddle.any(paddle.isinf(rhok)):
                rhok = 1000.0

            if len(sk_vec) > history_size:
                sk_vec.pop(0)
                yk_vec.pop(0)
                rhok_vec.pop(0)
            sk_vec.append(sk)
            yk_vec.append(yk)
            rhok_vec.append(rhok)
            xk = xk + sk
            g1 = g2
            k += 1

            g_norm = paddle.linalg.norm(g1, p=np.inf)
            if g_norm < tolerance_grad:
                is_converge = True
                break
            pk_norm = paddle.linalg.norm(pk, p=np.inf)
            if pk_norm < tolerance_change:
                is_converge = True
                break
            # when alpha=0, there is no chance to get xk change.
            if alpha == 0.:
                break
        return is_converge, num_func_calls, xk, value, g1
    else:
        is_converge = paddle.full(shape=[1], fill_value=False, dtype='bool')
        shape = initial_position.shape[0]
        k = paddle.full(shape=[1], fill_value=0, dtype='int64')
        history_size = paddle.assign(history_size)
        head = paddle.full(shape=[1], fill_value=1, dtype='int64')
        tail = paddle.full(shape=[1], fill_value=0, dtype='int64')
        done = paddle.full(shape=[1], fill_value=False, dtype='bool')
        sk_vec = paddle.zeros((history_size + 1, shape), dtype=dtype)
        yk_vec = paddle.zeros((history_size + 1, shape), dtype=dtype)
        rhok_vec = paddle.zeros((history_size + 1, 1), dtype=dtype)
        ai_vec = paddle.zeros((history_size + 1, 1), dtype=dtype)

        def cond(k, done, is_converge, num_func_calls, value, xk, g1, sk_vec,
                 yk_vec, rhok_vec, head, tail):
            return (k < max_iters) & ~done

        def body(k, done, is_converge, num_func_calls, value, xk, g1, sk_vec,
                 yk_vec, rhok_vec, head, tail):
            q = paddle.assign(g1)

            i = paddle.full(
                shape=[1],
                fill_value=(head - 1).mod(history_size),
                dtype='int64')

            def cond(i, q):
                return i != tail

            def body(i, q):
                ai_vec[i] = rhok_vec[i] * paddle.dot(sk_vec[i], q)
                q = q - ai_vec[i] * yk_vec[i]
                i = (i - 1).mod(history_size)
                return i, q

            paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[i, q])

            r = paddle.matmul(H0, q)

            i = paddle.full(shape=[1], fill_value=tail + 1, dtype='int64')

            def cond(i, r):
                return i != head

            def body(i, r):
                beta = rhok_vec[i] * paddle.dot(yk_vec[i], r)
                r = r + sk_vec[i] * (ai_vec[i] - beta)
                i = (i + 1).mod(history_size)
                return i, r

            paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[i, r])

            pk = -r

            if line_search_fn == 'strong_wolfe':
                alpha, value, g2, ls_func_calls = strong_wolfe(
                    f=objective_func,
                    xk=xk,
                    pk=pk,
                    initial_step_length=initial_step_length)
            else:
                raise NotImplementedError(
                    "Currently only support line_search_fn = 'strong_wolfe', but the specified is '{}'".
                    format(line_search_fn))
            paddle.assign(num_func_calls + ls_func_calls, num_func_calls)

            sk = alpha * pk
            yk = g2 - g1

            rhok = 1. / paddle.dot(yk, sk)

            def true_fn2(rhok):
                paddle.assign(1000.0, rhok)

            paddle.static.nn.cond(
                paddle.any(paddle.isinf(rhok)), lambda: true_fn2(rhok), None)

            sk_vec[head] = sk
            yk_vec[head] = yk
            rhok_vec[head] = rhok
            head = (head + 1) % history_size

            def true_fn(tail):
                paddle.assign(tail + 1, tail)

            paddle.static.nn.cond(head == tail, lambda: true_fn(tail), None)

            paddle.assign(xk + sk, xk)
            paddle.assign(g2, g1)
            k += 1

            gnorm = paddle.linalg.norm(g1, p=np.inf)
            pk_norm = paddle.linalg.norm(pk, p=np.inf)
            paddle.assign(done | (gnorm < tolerance_grad) |
                          (pk_norm < tolerance_change), done)
            paddle.assign(done, is_converge)

            paddle.assign(done | (alpha == 0.), done)

            return [
                k, done, is_converge, num_func_calls, value, xk, g1, sk_vec,
                yk_vec, rhok_vec, head, tail
            ]

        paddle.static.nn.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                k, done, is_converge, num_func_calls, value, xk, g1, sk_vec,
                yk_vec, rhok_vec, head, tail
            ])
        return is_converge, num_func_calls, xk, value, g1
