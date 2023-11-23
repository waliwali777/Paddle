/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
// common help func
#include "paddle/fluid/platform/device/gcu/equivalence_trans/utils.h"
// CHANNEL FIRST OPS
#include "paddle/fluid/platform/device/gcu/equivalence_trans/channel_first_ops/conv2d.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/channel_first_ops/conv3d.h"
// CHANNEL LAST OPS
#include "paddle/fluid/platform/device/gcu/equivalence_trans/channel_last_ops/conv2d.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/channel_last_ops/conv3d.h"
// INSENSITIVE OPS
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/accuracy.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/activation.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/adam.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/adamw.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/add_n.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/argmax.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/argmin.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/argsort.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/assign.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/assign_value.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/atan.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/batch_norm.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/bilinear_interp_v2.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/bmm.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/cast.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/check_finite_and_unscale.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/clip.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/concat.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/conv2d.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/conv3d.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/cos.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/cross_entropy.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/cumsum.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/dropout.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/elementwise_binary.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/elementwise_unary.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/embedding.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/equal.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/expand.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/expand_as.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/fill_constant.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/fill_zeros_like.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/flatten.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/flip.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/floor.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/full_like.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/gather.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/gather_nd.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/gelu.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/grid_sampler.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/huber_loss.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/increment.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/index_select.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/instance_norm.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/iou_similarity.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/isinf_v2.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/label_smooth.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/layer_norm.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/log.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/log_loss.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/log_softmax.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/logical_and.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/logical_not.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/masked_select.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/matmul_v2.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/maximum.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/mean.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/meshgrid.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/minimum.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/momentum.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/mul.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/nearest_interp.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/nearest_interp_v2.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/not_equal.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/one_hot.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/pool2d.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/prior_box.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/random.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/range.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/reduce_x.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/reshape.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/reverse.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/rmsprop.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/rnn.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/roi_align.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/roll.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/scale.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/scatter.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/set_value.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/shape.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/share_data.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/sigmoid_cross_entropy_with_logits.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/sign.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/size.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/slice.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/softmax.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/softmax_with_cross_entropy.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/split.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/sqrt.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/squared_l2_norm.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/squeeze.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/stack.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/strided_slice.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/tanh.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/tile.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/topk.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/transpose.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/tril_triu.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/unsqueeze.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/unstack.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/where.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/insensitive_ops/yolo_box.h"
