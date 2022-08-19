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

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/op_kernel_type.h"

namespace paddle {
namespace platform {

using vartype = paddle::framework::proto::VarType;
using pOpKernelType = paddle::framework::OpKernelType;
using XPUKernelSet =
    std::unordered_set<pOpKernelType, paddle::framework::OpKernelType::Hash>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& get_kl2_ops() {
  // KL1支持的op，通过op_name, data_type, place来索引
  static XPUOpMap s_xpu2_kernels{
      {"abs", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"abs_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"adamw", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"adam", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"arg_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"argsort",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"assign",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace())})},
      {"assign_value",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bmm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bmm_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bce_loss_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bce_loss", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"beam_search",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"beam_search_decode",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"bilinear_interp_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bilinear_interp_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"broadcast", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_allgather",
       XPUKernelSet({pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"c_allreduce_sum",
       XPUKernelSet({pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_identity",
       XPUKernelSet({pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"c_sync_comm_stream",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"cast",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"check_finite_and_unscale",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"clip", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"coalesce_tensor",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"concat_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"concat",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"conv2d_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"conv2d",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"conv2d_transpose_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"conv2d_transpose",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"deformable_conv_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"deformable_conv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"depthwise_conv2d_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"depthwise_conv2d",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"dropout_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"dropout",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_add_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_add",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_div_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_div",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_floordiv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_max",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_min_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_min",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mul_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_pow",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_sub_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_sub",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mod",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"empty",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT16, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace())})},
      {"equal",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"exp", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"expand_as_v2",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"expand_v2",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"fill_any_like",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"fill_constant",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT16, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::COMPLEX64, XPUPlace()),
                     pOpKernelType(vartype::COMPLEX128, XPUPlace())})},
      {"flatten2_grad",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten2",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten_contiguous_range_grad",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten_contiguous_range",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten_grad",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"flatten",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gather_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"gather_nd",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gather",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"gaussian_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gelu_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"gelu",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"generate_proposals_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"grad_add",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"greater_equal",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"greater_than",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"grid_sampler",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"hard_swish_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"hard_swish", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"huber_loss_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"huber_loss", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"iou_similarity",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"instance_norm",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"instance_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"label_smooth",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lars_momentum",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"layer_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"layer_norm",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"leaky_relu_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"leaky_relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"less_equal",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"less_than",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log_softmax", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log_softmax_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lookup_table_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lookup_table_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"masked_select",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"matmul_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"matmul_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"matmul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mean_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mean",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"merged_momentum",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mish_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mish", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"momentum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mul_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"nearest_interp_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"nearest_interp_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"not_equal",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"one_hot",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"one_hot_v2",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"p_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"p_norm_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"pool2d_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"pool2d",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"pow", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"pow_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"pow2_decay_with_linear_warmup",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"prior_box", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"range",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"reciprocal", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reciprocal_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"reduce_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_mean_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_mean", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_prod", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_sum_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_sum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu6", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu6_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"relu",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"reshape2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reshape2",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"resnet_unit",
       XPUKernelSet({pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"resnet_unit_grad",
       XPUKernelSet({pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"rmsprop", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"rnn", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"rnn_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"roi_align", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"roi_align_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"scale",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"scatter",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sampling_id",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace())})},
      {"sgd",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"sigmoid_cross_entropy_with_logits_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid_cross_entropy_with_logits",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"shape",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace())})},
      {"sigmoid", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sign", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"slice_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"slice",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"softmax",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"softmax_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"softmax_with_cross_entropy_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"softmax_with_cross_entropy",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"softplus", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softplus_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"split",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"square_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"square", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze2",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"stack",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"stack_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"sum",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"swish", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"swish_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"tanh_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"tanh",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"tril_triu",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"tril_triu_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"tile",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"transpose2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"truncated_gaussian_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"top_k",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"top_k_v2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"update_loss_scaling",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"uniform_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"unsqueeze2",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"unsqueeze_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"where_index",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"where",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},

      // AddMore
      {"sequence_conv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sequence_conv_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sequence_unpad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      // Fused op
      {"resnet_basic_block_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"resnet_basic_block",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
  };

  return s_xpu2_kernels;
}

}  // namespace platform
}  // namespace paddle
#endif
