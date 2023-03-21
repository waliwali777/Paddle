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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature AttentionFuseGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("fused_attention",
                         {"Y@grad",
                          "X",
                          "QKVW",
                          "QKVBias",
                          "QKVBiasOut",
                          "SrcMask",
                          "SrcMaskOut",
                          "OutLinearW",
                          "OutLinearBias",
                          "LnScale",
                          "LnBias",
                          "Ln2Scale",
                          "Ln2Bias",
                          "LnOut",
                          "LnMean",
                          "LnVariance",
                          "Ln2Mean",
                          "Ln2Variance",
                          "BiasDropoutResidualOut",
                          "QKVOut",
                          "TransposeOut2",
                          "QKOut",
                          "QKTVOut",
                          "SoftmaxOut",
                          "AttnDropoutMaskOut",
                          "AttnDropoutOut",
                          "FMHAOut",
                          "OutLinearOut",
                          "DropoutMaskOut"},
                         {"num_heads",
                          "transpose_qkv_wb",
                          "pre_layer_norm",
                          "epsilon",
                          "attn_dropout_rate",
                          "is_test",
                          "attn_dropout_fix_seed",
                          "attn_dropout_seed",
                          "attn_dropout_implementation",
                          "dropout_rate",
                          "dropout_fix_seed",
                          "dropout_seed",
                          "dropout_implementation",
                          "ln_epsilon",
                          "add_residual",
                          "ring_id"},
                         {
                             "QKVBias@grad",
                             "QKVBiasOut@grad",
                             "SrcMaskOut@grad",
                             "OutLinearBias@grad",
                             "LnScale@grad",
                             "LnBias@grad",
                             "Ln2Scale@grad",
                             "Ln2Bias@grad",
                             "X@grad",
                             "QKVW@grad",
                             "OutLinearW@grad",
                             "LnOut@grad",
                             "BiasDropoutResidualOut@grad",
                             "QKVOut@grad",
                             "QKTVOut@grad",
                             "TransposeOut2@grad",
                             "QKOut@grad",
                             "SoftmaxOut@grad",
                             "AttnDropoutOut@grad",
                             "FMHAOut@grad",
                             "OutLinearOut@grad",
                         });
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_attention_grad,
                           phi::AttentionFuseGradOpArgumentMapping);
