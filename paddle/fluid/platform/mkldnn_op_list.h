/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef PADDLE_WITH_MKLDNN

#include <unordered_set>

namespace paddle {
namespace platform {

static const std::unordered_set<std::string> mkldnn_black_list = {
    "abs",
    "abs_grad",
    "batch_norm",
    "batch_norm_grad",
    "clip",
    "clip_grad",
    "concat",
    "concat_grad",
    "data_norm",
    "data_norm_grad",
    "expand_v2",
    "expand_v2_grad",
    "fill_constant",
    "fusion_gru",
    "fusion_lstm",
    "gaussian_random",
    "log_softmax",
    "lrn",
    "lrn_grad",
    "matmul",
    "matmul_grad",
    "matmul_v2",
    "matmul_v2_grad",
    "scale",
    "shape",
    "shuffle_channel",
    "stack",
    "transpose",
    "transpose_grad",
    "transpose2_grad",
    "elementwise_add",
    "elementwise_add_grad",
    "elementwise_sub",
    "elementwise_sub_grad",
    "elementwise_mul",
    "elementwise_mul_grad",
    "elementwise_div",
    "elementwise_div_grad",
    "gelu",
    "gelu_grad",
    "prelu",
    "prelu_grad",
    "nearest_interp",
    "bilinear_interp",
    "nearest_interp_v2",
    "bilinear_interp_v2",
    // return mkldnn ExpectedKernelType directly
    "quantize",  // need mkldnn kernel, trigger assert error
    "requantize",
    "dequantize",
    // activation mkldnn operator
    "soft_relu",
    "soft_relu_grad",
    "cos",
    "cos_grad",
    "tan",
    "tan_grad",
    "acos",
    "acos_grad",
    "sin",
    "sin_grad",
    "asin",
    "asin_grad",
    "atan",
    "atan_grad",
    "sinh",
    "sinh_grad",
    "cosh",
    "cosh_grad",
    "asinh",
    "asinh_grad",
    "acosh",
    "acosh_grad",
    "atanh",
    "atanh_grad",
    "brelu",
    "brelu_grad",
    "thresholded_relu",
    "thresholded_relu_grad",
    "relu6",
    "relu6_grad",
    "hard_shrink",
    "hard_shrink_grad",
    "softshrink",
    "softshrink_grad",
    "tanh_shrink",
    "tanh_shrink_grad",
    "silu",
    "silu_grad",
    "softsign",
    "softsign_grad",
    "hard_sigmoid",
    "hard_sigmoid_grad",
    "logsigmoid",
    "logsigmoid_grad",
    "expm1",
    "expm1_grad",
    "softplus",
    "softplus_grad",
    "mish",
    "mish_grad",
    "stanh",
    "stanh_grad",
    "reciprocal",
    "reciprocal_grad",
    "log2",
    "log2_grad",
    "log10",
    "log10_grad",
    "log1p",
    "log1p_grad",
    "hard_swish",
    "hard_swish_grad",
    "swish",
    "swish_grad",
    "round",
    "round_grad",
    "floor",
    "floor_grad",
    "ceil",
    "ceil_grad",
    "sigmoid",
    "sigmoid_grad",
    "sigmoid_grad_grad",
    "sigmoid_triple_grad",
    "tanh",
    "tanh_grad",
    "tanh_grad_grad",
    "tanh_triple_grad",
    "relu",
    "relu_grad",
    "relu_grad_grad",
    "leaky_relu",
    "leaky_relu_grad",
    "leaky_relu_grad_grad",
    "elu",
    "elu_grad",
    "elu_grad_grad",
    "celu",
    "celu_grad",
    "celu_grad_grad",
    "logit",
    "logit_grad",
    "sqrt",
    "sqrt_grad",
    "sqrt_grad_grad",
    "rsqrt",
    "rsqrt_grad",
    "rsqrt_grad_grad",
    "square",
    "square_grad",
    "square_grad_grad",
    "pow",
    "pow_grad",
    "exp",
    "exp_grad",
    "log",
    "log_grad",
    "log_grad_grad",
};

static const std::unordered_set<std::string> mkldnn_white_list = {
    "cast",
    "transfer_dtype",
    "conv2d_transpose",
    "depthwise_conv2d_transpose",
    "conv3d_transpose",
    "layer_norm",
    "pad2d",
    "pad3d",
    "pool2d",
    "pool2d_grad",
    "pool2d_double_grad",
    "pool3d",
    "pool3d_grad",
    "slice",
    "slice_grad",
    "split",
    "sum",
    "sgd",
    "softmax",
    "softmax_grad",
    // NOTE(jiahy0825): squeeze MKLDNN kernel are disabled
    // (https://github.com/PaddlePaddle/Paddle/pull/35781). If these MKLDNN
    // kernels and codes are deleted in the future, attributes `use_mkldnn`
    // should be removed from function declaration
    "squeeze",
    "squeeze_grad",
    "squeeze2",
    "squeeze2_grad",
    // NOTE(jiahy0825): reshape and flatten has attribute use_mkldnn, but they
    // didn't change the ExpectedKernelType of tensor
    "reshape",
    "reshape_grad",
    "reshape2",  // need mkldnn kernel, trigger assert error
    "reshape2_grad",
    "flatten",
    "flatten_grad",
    "flatten2",
    "flatten2_grad",
    // NOTE(jiahy0825): After fixing GetExpectedKernelType in ReduceOp, reduce
    // series hard code can be deleted together.
    "frobenius_norm",
    "reduce_amax",
    "reduce_amin",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    "frobenius_norm_grad",
    "reduce_amax_grad",
    "reduce_amin_grad",
    "reduce_max_grad",
    "reduce_mean_grad",
    "reduce_min_grad",
    "reduce_prod_grad",
    "reduce_sum_grad",
    // NOTE(jiahy0825): Below ops register kernel with customized_type_value, we
    // need to analysis and solve them one-by-one.
    "conv2d",
    "conv2d_grad",
    "depthwise_conv2d",
    "depthwise_conv2d_grad",
    "conv3d",
    "conv3d_grad",
    "prior_box",
    "fc",
    "mul",
    "mul_grad",
    "transpose2"};

inline bool in_mkldnn_black_list(const std::string& op_name) {
  return mkldnn_black_list.find(op_name) != mkldnn_black_list.end();
}

inline bool in_mkldnn_white_list(const std::string& op_name) {
  return mkldnn_white_list.find(op_name) != mkldnn_white_list.end();
}

}  // namespace platform
}  // namespace paddle
#endif
