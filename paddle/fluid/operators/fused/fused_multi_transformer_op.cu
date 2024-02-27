/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_multi_transformer_op.cu.h"

#include "paddle/fluid/framework/op_registry.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"

namespace phi {
namespace fusion {

#if CUDA_VERSION >= 11060  // Use cublasLt to fuse FFN operation.

template <typename T, typename Context>
void FusedMultiTransformerKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const std::vector<const DenseTensor *> &ln_scales,
    const std::vector<const DenseTensor *> &ln_biases,
    const std::vector<const DenseTensor *> &qkv_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &qkv_biases,
    const paddle::optional<std::vector<const DenseTensor *>> &cache_kvs,
    const paddle::optional<std::vector<const DenseTensor *>> &pre_caches,
    const paddle::optional<DenseTensor> &rotary_tensor,
    const paddle::optional<DenseTensor> &time_step,
    const paddle::optional<DenseTensor> &seq_lengths,
    const paddle::optional<DenseTensor> &src_mask,
    const std::vector<const DenseTensor *> &out_linear_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &out_linear_biases,
    const std::vector<const DenseTensor *> &ffn_ln_scales,
    const std::vector<const DenseTensor *> &ffn_ln_biases,
    const std::vector<const DenseTensor *> &ffn1_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn1_biases,
    const std::vector<const DenseTensor *> &ffn2_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn2_biases,
    bool pre_layer_norm,
    float epsilon,
    float dropout_rate,
    int rotary_emb_dims,
    bool is_test,
    const std::string &dropout_implementation,
    const std::string &act_method,
    bool trans_qkvw,
    int ring_id,
    std::vector<DenseTensor *> cache_kv_outs,
    DenseTensor *out) {
  VLOG(1) << "start kernel1";
  if (cache_kvs) {
    for (size_t i = 0; i < cache_kv_outs.size(); i++) {
      *(cache_kv_outs[i]) = *(cache_kvs.get()[i]);
    }
  }
  using U = phi::funcs::LayerNormParamType<T>;

  auto *rotary_tensor_t = rotary_tensor.get_ptr();
  auto *seq_lengths_t = seq_lengths.get_ptr();
  auto *src_mask_t = src_mask.get_ptr();
  auto *time_step_t = time_step.get_ptr();

  const auto input_x_dims = x.dims();
  int bsz = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];
  int bsz_seq = bsz * seq_len;
  bool remove_padding = false;
  if (seq_lengths_t) {
    remove_padding = true;
  }
  phi::DenseTensor d_token_tensor;
  phi::DenseTensor padding_offset_tensor;
  phi::DenseTensor x_remove_padding;
  bool encoder_remove_padding = (remove_padding && !time_step_t);
  int token_num = 0;

  // remove padding in encoder
  if (encoder_remove_padding) {
    // just for encoder
    d_token_tensor.Resize({1});
    auto *d_token_num = dev_ctx.template Alloc<int>(
        &d_token_tensor, d_token_tensor.numel() * sizeof(int));
    // alloc the max size of padding_offset_tensor
    padding_offset_tensor.Resize({bsz_seq});
    dev_ctx.template Alloc<int>(&padding_offset_tensor,
                                padding_offset_tensor.numel() * sizeof(int));
    InvokeGetPaddingOffset(dev_ctx,
                           &token_num,
                           d_token_num,
                           padding_offset_tensor.data<int>(),
                           seq_lengths_t->data<int>(),
                           bsz,
                           seq_len);
    padding_offset_tensor.Resize({token_num});
    x_remove_padding.Resize({token_num, dim_embed});
    dev_ctx.template Alloc<T>(&x_remove_padding,
                              x_remove_padding.numel() * sizeof(T));
    InvokeRemovePadding(dev_ctx,
                        x_remove_padding.data<T>(),
                        x.data<T>(),
                        padding_offset_tensor.data<int>(),
                        token_num,
                        dim_embed);
  } else {
    token_num = bsz_seq;
  }
  auto *padding_offset_data =
      encoder_remove_padding ? padding_offset_tensor.data<int>() : nullptr;

  auto ln_compute = AttnLayerNorm<T>(dev_ctx, epsilon, token_num, dim_embed);
  phi::DenseTensor ln_mean, ln_var;
  ln_mean.Resize({token_num});
  auto *ln_mean_data =
      dev_ctx.template Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
  ln_var.Resize({token_num});
  auto *ln_var_data =
      dev_ctx.template Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

  // 2. qkv
  // x: qkv's input [batch_size, seq_len, dim_embed]
  // y: qkv's weight: [3, num_head, dim_head, dim_embed]
  const auto qkv_w_dims = qkv_weights[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
  int hidden_size = num_head * dim_head;
  int output_size = 3 * hidden_size;
  int input_size = dim_embed;

  bool compute_bias =
      qkv_biases && !qkv_biases.get().empty() && time_step_t == nullptr;
  // (transA, transB, compute_bias) = (false, trans_qkvw, false)
  // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we set
  // compute_bias as false.
  auto qkv_compute = phi::fusion::AttnMatMul<T>(dev_ctx,
                                                false,
                                                trans_qkvw,
                                                token_num,
                                                output_size,
                                                input_size,
                                                /*compute_bias=*/false);

  phi::DenseTensor qkv_out;
  qkv_out.Resize({token_num, 3, num_head, dim_head});
  auto *qkv_out_data =
      dev_ctx.template Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

  // 3. fmha
  AttnDropoutParam attn_param(
      true, "upscale_in_train", 0.0, true, true, 0, nullptr);
  auto fmha_compute =
      FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
  int cache_offset = 0;
  if (pre_caches && pre_caches.get().size() > 0) {
    cache_offset = pre_caches.get()[0]->dims()[3];
  }

  auto out_seq_len = seq_len;
  if (time_step_t) {
    PADDLE_ENFORCE_EQ(time_step_t->place(),
                      phi::CPUPlace(),
                      phi::errors::PreconditionNotMet(
                          "The place of input(TimeStep) must be CPUPlace."));
    // cache_seq_len
    int time_step_value = time_step_t->data<int>()[0];
    PADDLE_ENFORCE_GT(time_step_value,
                      0,
                      phi::errors::PreconditionNotMet(
                          "The value of time_step_t must > 0, but now is %d",
                          time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        phi::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
    out_seq_len += time_step_value;
  } else {
    out_seq_len += cache_offset;
  }

  phi::DenseTensor q_transpose_out, kv_transpose_out, qk_out;
  q_transpose_out.Resize({bsz, num_head, seq_len, dim_head});
  auto *q_transpose_out_data = dev_ctx.template Alloc<T>(
      &q_transpose_out, q_transpose_out.numel() * sizeof(T));

  kv_transpose_out.Resize({2, bsz, num_head, seq_len, dim_head});
  auto *kv_transpose_out_data = dev_ctx.template Alloc<T>(
      &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

  qk_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *qk_out_data =
      dev_ctx.template Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

  phi::DenseTensor src_mask_out;
  if (cache_offset > 0) {
    src_mask_out.Resize({bsz, num_head, seq_len, out_seq_len});
    auto *src_mask_out_data = dev_ctx.template Alloc<T>(
        &src_mask_out, src_mask_out.numel() * sizeof(T));
  }

  // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
  phi::DenseTensor pre_cache_kv_out;
  if (cache_offset > 0) {
    pre_cache_kv_out.Resize(
        {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
    auto *pre_cache_kv_out_data = dev_ctx.template Alloc<T>(
        &pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
  }

  phi::DenseTensor softmax_out;
  phi::DenseTensor attn_dropout_mask_out, attn_dropout_out;
  phi::DenseTensor qktv_out, fmha_out;
  softmax_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *softmax_out_data =
      dev_ctx.template Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

  attn_dropout_mask_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *attn_dropout_mask_out_data = dev_ctx.template Alloc<T>(
      &attn_dropout_mask_out, attn_dropout_mask_out.numel() * sizeof(T));
  attn_dropout_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *attn_dropout_data_data = dev_ctx.template Alloc<T>(
      &attn_dropout_out, attn_dropout_out.numel() * sizeof(T));

  qktv_out.Resize({bsz, num_head, seq_len, dim_head});
  auto *qktv_out_data =
      dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
  fmha_out.Resize({bsz, seq_len, num_head, dim_head});
  auto *fmha_out_data =
      dev_ctx.template Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

  // (transA, transB, compute_bias) = (false, false, false)
  auto out_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, token_num, dim_embed, hidden_size, false);

  // 5. ln(residual + bias)
  DropoutParam dropout_param2(true, 0, true, true, 0.0, nullptr, 0);
  FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
      dev_ctx, token_num, dim_embed, dropout_param2, epsilon);
  phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
  T *bias_dropout_residual_out_data = nullptr;
  if (pre_layer_norm) {
    bias_dropout_residual_out.Resize({token_num, dim_embed});
    bias_dropout_residual_out_data = dev_ctx.template Alloc<T>(
        &bias_dropout_residual_out,
        bias_dropout_residual_out.numel() * sizeof(T));
  }
  dropout_mask_out.Resize({token_num, dim_embed});
  auto *dropout_mask_out_data = dev_ctx.template Alloc<uint8_t>(
      &dropout_mask_out, dropout_mask_out.numel() * sizeof(uint8_t));

  // 6. ffn1 matmul + act + bias
  auto ffn1_weight_dim = ffn1_weights[0]->dims();

  int dim_ffn = ffn1_weight_dim[1];

  auto ffn1_cublas_linear = CublasFusedMLP<T>(dev_ctx);
  const phi::DDim ffn1_input_shape({token_num, dim_embed});
  ffn1_cublas_linear.Setup(ffn1_input_shape, ffn1_weight_dim, false, false);

  phi::DenseTensor ffn1_out;
  ffn1_out.Resize({token_num, dim_ffn});
  auto *ffn1_out_data =
      dev_ctx.template Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

  // 7. ffn2 matmul + bias + residual.
  auto ffn2_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, token_num, dim_embed, dim_ffn, false);

  // 8. ffn2 Layernorm residual bias
  DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
  FusedDropoutLayerNormHelper<T, uint8_t> ffn2_fused_dropout_helper(
      dev_ctx, token_num, dim_embed, ffn2_dropout_param, epsilon);

  // calc
  auto *from_data = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  phi::DenseTensor *from_tensor = out;
  phi::DenseTensor tmp_out, tmp_out_rm_padding;
  tmp_out.Resize({token_num, dim_embed});
  if (encoder_remove_padding) {
    tmp_out_rm_padding.Resize({token_num, dim_embed});
    auto *tmp_out_rm_padding_data = dev_ctx.template Alloc<T>(
        &tmp_out_rm_padding, tmp_out_rm_padding.numel() * sizeof(T));
  }
  auto *tmp_out_data =
      dev_ctx.template Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

  const T *x_data;
  if (encoder_remove_padding) {
    x_data = x_remove_padding.data<T>();
  } else {
    x_data = x.data<T>();
  }
  phi::DenseTensor *buf0 = nullptr;
  phi::DenseTensor *buf1 = nullptr;

  // step0:  x   --> buf1
  // step1: buf1 --> buf0
  // step2: buf0 --> buf1
  int layers = qkv_weights.size();
  if (encoder_remove_padding) {
    // In the case of variable lengths, the padding needs to be rebuilt
    // eventually. So buf0 and buf1 do not need to be changed according to the
    // pre_layer_norm and the number of layers.
    buf0 = &tmp_out;
    buf1 = &tmp_out_rm_padding;
  } else {
    if (pre_layer_norm) {
      if (layers & 1) {
        // odd, set buf1 as out
        buf0 = &tmp_out;
        buf1 = out;
      } else {
        // even, set buf0 as out
        buf0 = out;
        buf1 = &tmp_out;
      }
    } else {
      buf0 = &tmp_out;
      buf1 = out;
    }
  }

  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    if (i == 0 && pre_layer_norm) {
      auto *ln_scale_data = ln_scales[i]->data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      // TODO(wangxi): can remove mean var in inference
      ln_compute.ComputeForward(x_data,
                                ln_scale_data,
                                ln_bias_data,
                                buf1->data<T>(),
                                ln_mean_data,
                                ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step1";
#endif

    // step2. qkv
    const phi::DenseTensor *qkv_bias =
        qkv_biases && !qkv_biases.get().empty() ? qkv_biases.get()[i] : nullptr;
    // NOTE: in decoder stage, bias is fused in fmha
    const phi::DenseTensor *bias = time_step_t ? nullptr : qkv_bias;
    if (!pre_layer_norm && i == 0) {
      const phi::DenseTensor *tmp_input_x =
          (encoder_remove_padding) ? &x_remove_padding : &x;
      qkv_compute.ComputeForward(
          qkv_weights[i], tmp_input_x, bias, &qkv_out, &qkv_out);
    } else {
      qkv_compute.ComputeForward(
          qkv_weights[i], buf1, bias, &qkv_out, &qkv_out);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step2";
#endif

    // step3. fmha
    const phi::DenseTensor *cache_kv =
        cache_kvs && cache_kvs.get().size() > 0 ? cache_kvs.get()[i] : nullptr;
    phi::DenseTensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

    if (time_step_t) {  // generation decoder stage
      // [2, batch_size, num_head, max_seq_len, head_size]
      int max_seq_len = cache_kv->dims()[3];
      fmha<T>(dev_ctx,
              qkv_out,
              *qkv_bias,
              *src_mask_t,
              seq_lengths_t,
              rotary_tensor_t,
              cache_kv_out,
              &fmha_out,
              bsz,
              max_seq_len,
              num_head,
              dim_head,
              time_step_t->data<int>()[0],
              rotary_emb_dims,
              1. / std::sqrt(dim_head));
    } else if (cache_kv_out) {  // generation context stage
      const phi::DenseTensor *pre_cache_kv_tensor =
          pre_caches && pre_caches.get().size() > 0 ? pre_caches.get()[i]
                                                    : nullptr;
      phi::DenseTensor *pre_cache_kv_out_tmp =
          cache_offset > 0 ? &pre_cache_kv_out : nullptr;
      phi::DenseTensor *src_mask_tmp =
          cache_offset > 0 ? &src_mask_out : nullptr;
      qkv_bias_add_transpose_split<T>(dev_ctx,
                                      q_transpose_out_data,
                                      kv_transpose_out_data,
                                      qkv_out_data,
                                      qkv_bias->data<T>(),
                                      padding_offset_data,
                                      token_num,
                                      bsz,
                                      num_head,
                                      seq_len,
                                      dim_head,
                                      compute_bias);
      // q_transpose_out_data [bs, head_num, seq_len, dim_head]
      // kv_transpose_out_data [2, bs, head_num, seq_len, dim_head]
      if (rotary_emb_dims != 0) {
        auto *rotary_emb_data = rotary_tensor_t->data<T>();
        const int *sequence_lengths_data =
            encoder_remove_padding ? seq_lengths_t->data<int>() : nullptr;
        rotary_qk(dev_ctx,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  rotary_emb_data,
                  sequence_lengths_data,
                  rotary_emb_dims,
                  bsz,
                  num_head,
                  seq_len,
                  dim_head);
      }

      phi::DenseTensor *tmp_padding_offset_tensor =
          encoder_remove_padding ? &padding_offset_tensor : nullptr;
      fmha_compute.ComputeForwardWithoutTranspose(pre_cache_kv_tensor,
                                                  src_mask_t,
                                                  tmp_padding_offset_tensor,
                                                  &q_transpose_out,
                                                  &kv_transpose_out,
                                                  pre_cache_kv_out_tmp,
                                                  &qk_out,
                                                  src_mask_tmp,
                                                  &softmax_out,
                                                  &attn_dropout_mask_out,
                                                  &attn_dropout_out,
                                                  &qktv_out,
                                                  &fmha_out,
                                                  token_num);
      const T *k_ptr = nullptr;
      const T *v_ptr = nullptr;

      if (cache_offset > 0) {
        // [2, bsz, num_head, cache_offset + seq_len, head_dim]
        const T *kv_data = pre_cache_kv_out.data<T>();
        k_ptr = kv_data;
        int64_t k_size = bsz * num_head * (seq_len + cache_offset) * dim_head;
        v_ptr = k_ptr + k_size;
      } else {
        // [3, bsz, num_head, seq_len, head_dim]
        int64_t k_size = bsz * seq_len * num_head * dim_head;
        const T *q_ptr = q_transpose_out_data;
        k_ptr = kv_transpose_out_data;
        v_ptr = k_ptr + k_size;
      }

      // [2, bsz, num_head, max_seq_len, head_dim]
      int max_seq_len = cache_kv_out->dims()[3];
      T *cache_kv_data = cache_kv_out->data<T>();
      int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

      T *cache_k_ptr = cache_kv_data;
      T *cache_v_ptr = cache_kv_data + cache_k_size;

      const int seq_len_tmp = seq_len + cache_offset;
      write_cache_kv<T>(dev_ctx,
                        cache_k_ptr,
                        cache_v_ptr,
                        k_ptr,
                        v_ptr,
                        bsz,
                        num_head,
                        seq_len_tmp,
                        max_seq_len,
                        dim_head);
    } else {  // not generation
      // TODO(wangxi): can remove dropout in inference
      qkv_bias_add_transpose_split<T>(dev_ctx,
                                      q_transpose_out_data,
                                      kv_transpose_out_data,
                                      qkv_out_data,
                                      qkv_bias->data<T>(),
                                      padding_offset_data,
                                      token_num,
                                      bsz,
                                      num_head,
                                      seq_len,
                                      dim_head,
                                      compute_bias);

      // q_transpose_out_data [bs, head_num, seq_len, dim_head]
      // kv_transpose_out_data [2, bs, head_num, seq_len, dim_head]
      if (rotary_emb_dims != 0) {
        auto *rotary_emb_data = rotary_tensor_t->data<T>();
        const int *sequence_lengths_data =
            encoder_remove_padding ? seq_lengths_t->data<int>() : nullptr;
        rotary_qk(dev_ctx,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  rotary_emb_data,
                  sequence_lengths_data,
                  rotary_emb_dims,
                  bsz,
                  num_head,
                  seq_len,
                  dim_head);
      }

      phi::DenseTensor *tmp_padding_offset_tensor =
          encoder_remove_padding ? &padding_offset_tensor : nullptr;
      fmha_compute.ComputeForwardWithoutTranspose(cache_kv,
                                                  src_mask_t,
                                                  tmp_padding_offset_tensor,
                                                  &q_transpose_out,
                                                  &kv_transpose_out,
                                                  cache_kv_out,
                                                  &qk_out,
                                                  nullptr,
                                                  &softmax_out,
                                                  &attn_dropout_mask_out,
                                                  &attn_dropout_out,
                                                  &qktv_out,
                                                  &fmha_out,
                                                  token_num);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step3";
#endif

    if (pre_layer_norm) {
      out_linear_compute.ComputeForward(
          out_linear_weights[i], &fmha_out, nullptr, buf1, nullptr);
      AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
    } else {
      out_linear_compute.ComputeForward(
          out_linear_weights[i], &fmha_out, nullptr, buf0, nullptr);
      AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step4";
#endif

    // step5. ln(residual + dropout(input + bias))
    if (pre_layer_norm) {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases.get()[i]->data<T>();

      // inplace
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf1->data<T>(),
          x_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    } else {
      auto *ln_scale_data = ln_scales[i]->data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases.get()[i]->data<T>();
      auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf0->data<T>(),
          residual_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step5";
#endif

    // step6. ffn matmul1
    ffn1_cublas_linear.ComputeForward(buf1,
                                      ffn1_weights[i],
                                      ffn1_biases.get()[i],
                                      nullptr,
                                      &ffn1_out,
                                      act_method);

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step6";
#endif

    // step7. ffn2 matmul
    if (pre_layer_norm) {
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_out, nullptr, buf1, nullptr);
    } else {
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_out, nullptr, buf0, nullptr);
    }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step7";
#endif

    if (pre_layer_norm) {
      AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
    } else {
      AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step7.1";
#endif

    // step8. layer norm + bias_add + residual
    if (pre_layer_norm) {
      // TODO(wangxi): remove dropout mask in inference
      if (i < layers - 1) {
        auto *ln_scale_data = ln_scales[i + 1]->data<U>();
        auto *ln_bias_data = ln_biases[i + 1]->data<U>();
        ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
            dev_ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            ffn2_biases.get()[i]->data<T>(),
            ln_scale_data,
            ln_bias_data,
            buf1->data<T>(),
            dropout_mask_out_data,
            buf0->data<T>(),
            ln_mean_data,
            ln_var_data);
      } else {
        ffn2_fused_dropout_helper.ResidualDropoutBias(
            dev_ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            ffn2_biases.get()[i]->data<T>(),
            buf1->data<T>(),
            dropout_mask_out_data);
      }
    } else {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf0->data<T>(),
          buf1->data<T>(),
          ffn2_biases.get()[i]->data<T>(),
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step8";
#endif
    if (pre_layer_norm) {
      x_data = buf1->data<T>();
      std::swap(buf0, buf1);
    }
  }
  if (encoder_remove_padding) {
    if (pre_layer_norm) {
      InvokeRebuildPadding(dev_ctx,
                           from_data,
                           buf0->data<T>(),
                           padding_offset_data,
                           token_num,
                           dim_embed);
    } else {
      InvokeRebuildPadding(dev_ctx,
                           from_data,
                           buf1->data<T>(),
                           padding_offset_data,
                           token_num,
                           dim_embed);
    }
  }
}

#else

template <typename T, typename Context>
void FusedMultiTransformerKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const std::vector<const DenseTensor *> &ln_scales,
    const std::vector<const DenseTensor *> &ln_biases,
    const std::vector<const DenseTensor *> &qkv_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &qkv_biases,
    const paddle::optional<std::vector<const DenseTensor *>> &cache_kvs,
    const paddle::optional<std::vector<const DenseTensor *>> &pre_caches,
    const paddle::optional<DenseTensor> &rotary_tensor,
    const paddle::optional<DenseTensor> &time_step,
    const paddle::optional<DenseTensor> &seq_lengths,
    const paddle::optional<DenseTensor> &src_mask,
    const std::vector<const DenseTensor *> &out_linear_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &out_linear_biases,
    const std::vector<const DenseTensor *> &ffn_ln_scales,
    const std::vector<const DenseTensor *> &ffn_ln_biases,
    const std::vector<const DenseTensor *> &ffn1_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn1_biases,
    const std::vector<const DenseTensor *> &ffn2_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn2_biases,
    bool pre_layer_norm,
    float epsilon,
    float dropout_rate,
    int rotary_emb_dims,
    bool is_test,
    const std::string &dropout_implementation,
    const std::string &act_method,
    bool trans_qkvw,
    int ring_id,
    std::vector<DenseTensor *> cache_kv_outs,
    DenseTensor *out) {
  VLOG(1) << "start kernel2";
  VLOG(1) << "cache_kv_outs.size()" << cache_kv_outs.size();
  VLOG(1) << "cache_kvs->size()" << cache_kvs->size();
  if (cache_kvs) {
    for (size_t i = 0; i < cache_kv_outs.size(); i++) {
      *(cache_kv_outs[i]) = *(cache_kvs.get()[i]);
    }
  }
  using U = phi::funcs::LayerNormParamType<T>;
  auto *rotary_tensor_t = rotary_tensor.get_ptr();
  auto *seq_lengths_t = seq_lengths.get_ptr();
  auto *src_mask_t = src_mask.get_ptr();
  auto *time_step_t = time_step.get_ptr();

  VLOG(1) << "1111111";
  // 0. input
  const auto input_x_dims = x.dims();
  int bsz = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];
  int bsz_seq = bsz * seq_len;
  bool remove_padding = false;
  if (seq_lengths_t) {
    remove_padding = true;
  }
  phi::DenseTensor d_token_tensor;
  phi::DenseTensor padding_offset_tensor;
  phi::DenseTensor x_remove_padding;
  bool encoder_remove_padding = (remove_padding && !time_step_t);
  int token_num = 0;

  VLOG(1) << "222222222222";
  // remove padding in encoder
  if (encoder_remove_padding) {
    // just for encoder
    d_token_tensor.Resize({1});
    auto *d_token_num = dev_ctx.template Alloc<int>(
        &d_token_tensor, d_token_tensor.numel() * sizeof(int));
    // alloc the max size of padding_offset_tensor
    padding_offset_tensor.Resize({bsz_seq});
    dev_ctx.template Alloc<int>(&padding_offset_tensor,
                                padding_offset_tensor.numel() * sizeof(int));
    InvokeGetPaddingOffset(dev_ctx,
                           &token_num,
                           d_token_num,
                           padding_offset_tensor.data<int>(),
                           seq_lengths_t->data<int>(),
                           bsz,
                           seq_len);
    padding_offset_tensor.Resize({token_num});
    x_remove_padding.Resize({token_num, dim_embed});
    dev_ctx.template Alloc<T>(&x_remove_padding,
                              x_remove_padding.numel() * sizeof(T));
    InvokeRemovePadding(dev_ctx,
                        x_remove_padding.data<T>(),
                        x.data<T>(),
                        padding_offset_tensor.data<int>(),
                        token_num,
                        dim_embed);
  } else {
    token_num = bsz_seq;
  }
  auto *padding_offset_data =
      encoder_remove_padding ? padding_offset_tensor.data<int>() : nullptr;

  // 1. layer norm

  auto ln_compute = AttnLayerNorm<T>(dev_ctx, epsilon, token_num, dim_embed);
  phi::DenseTensor ln_mean, ln_var;
  ln_mean.Resize({token_num});
  auto *ln_mean_data =
      dev_ctx.template Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
  ln_var.Resize({token_num});
  auto *ln_var_data =
      dev_ctx.template Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

  // 2. qkv
  // x: qkv's input [batch_size, seq_len, dim_embed]
  // y: qkv's weight: [3, num_head, dim_head, dim_embed]
  const auto qkv_w_dims = qkv_weights[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
  int hidden_size = num_head * dim_head;
  int output_size = 3 * hidden_size;
  int input_size = dim_embed;

  bool compute_bias =
      qkv_biases && !qkv_biases.get().empty() && time_step_t == nullptr;
  // (transA, transB, compute_bias) = (false, trans_qkvw, false)
  // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we
  // set compute_bias as false.
  auto qkv_compute = phi::fusion::AttnMatMul<T>(dev_ctx,
                                                false,
                                                trans_qkvw,
                                                token_num,
                                                output_size,
                                                input_size,
                                                /*compute_bias=*/false);

  phi::DenseTensor qkv_out;
  qkv_out.Resize({token_num, 3, num_head, dim_head});
  auto *qkv_out_data =
      dev_ctx.template Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

  VLOG(1) << "33333333333";
  // 3. fmha
  AttnDropoutParam attn_param(
      true, "upscale_in_train", 0.0, true, true, 0, nullptr);
  auto fmha_compute =
      FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
  int cache_offset = 0;
  if (pre_caches && pre_caches.get().size() > 0) {
    cache_offset = pre_caches.get()[0]->dims()[3];
  }

  auto out_seq_len = seq_len;
  if (time_step_t) {
    PADDLE_ENFORCE_EQ(time_step_t->place(),
                      phi::CPUPlace(),
                      phi::errors::PreconditionNotMet(
                          "The place of input(TimeStep) must be CPUPlace."));
    // cache_seq_len
    int time_step_value = time_step_t->data<int>()[0];
    PADDLE_ENFORCE_GT(time_step_value,
                      0,
                      phi::errors::PreconditionNotMet(
                          "The value of time_step_t must > 0, but now is %d",
                          time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        phi::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
    out_seq_len += time_step_value;
  } else {
    out_seq_len += cache_offset;
  }

  phi::DenseTensor q_transpose_out, kv_transpose_out, qk_out;
  q_transpose_out.Resize({bsz, num_head, seq_len, dim_head});
  auto *q_transpose_out_data = dev_ctx.template Alloc<T>(
      &q_transpose_out, q_transpose_out.numel() * sizeof(T));

  kv_transpose_out.Resize({2, bsz, num_head, seq_len, dim_head});
  auto *kv_transpose_out_data = dev_ctx.template Alloc<T>(
      &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

  qk_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *qk_out_data =
      dev_ctx.template Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

  phi::DenseTensor src_mask_out;
  if (cache_offset > 0) {
    src_mask_out.Resize({bsz, num_head, seq_len, out_seq_len});
    auto *src_mask_out_data = dev_ctx.template Alloc<T>(
        &src_mask_out, src_mask_out.numel() * sizeof(T));
  }

  VLOG(1) << "444444444444444";
  // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
  phi::DenseTensor pre_cache_kv_out;
  if (cache_offset > 0) {
    pre_cache_kv_out.Resize(
        {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
    auto *pre_cache_kv_out_data = dev_ctx.template Alloc<T>(
        &pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
  }

  phi::DenseTensor softmax_out;
  phi::DenseTensor attn_dropout_mask_out, attn_dropout_out;
  phi::DenseTensor qktv_out, fmha_out;
  softmax_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *softmax_out_data =
      dev_ctx.template Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

  attn_dropout_mask_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *attn_dropout_mask_out_data = dev_ctx.template Alloc<T>(
      &attn_dropout_mask_out, attn_dropout_mask_out.numel() * sizeof(T));
  attn_dropout_out.Resize({bsz, num_head, seq_len, out_seq_len});
  auto *attn_dropout_data_data = dev_ctx.template Alloc<T>(
      &attn_dropout_out, attn_dropout_out.numel() * sizeof(T));

  qktv_out.Resize({bsz, num_head, seq_len, dim_head});
  auto *qktv_out_data =
      dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
  fmha_out.Resize({bsz, seq_len, num_head, dim_head});
  auto *fmha_out_data =
      dev_ctx.template Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

  // 4. out_linear
  // (transA, transB, compute_bias) = (false, false, false)
  auto out_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, token_num, dim_embed, hidden_size, false);

  // 5. ln(residual + bias)
  DropoutParam dropout_param2(true, 0, true, true, 0.0, nullptr, 0);
  FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
      dev_ctx, token_num, dim_embed, dropout_param2, epsilon);
  phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
  T *bias_dropout_residual_out_data = nullptr;
  if (pre_layer_norm) {
    bias_dropout_residual_out.Resize({token_num, dim_embed});
    bias_dropout_residual_out_data = dev_ctx.template Alloc<T>(
        &bias_dropout_residual_out,
        bias_dropout_residual_out.numel() * sizeof(T));
  }
  dropout_mask_out.Resize({token_num, dim_embed});
  auto *dropout_mask_out_data = dev_ctx.template Alloc<uint8_t>(
      &dropout_mask_out, dropout_mask_out.numel() * sizeof(uint8_t));

  // 6. ffn matmul1
  auto ffn1_weight_dim = ffn1_weights[0]->dims();

  int dim_ffn = ffn1_weight_dim[1];
  auto ffn1_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, token_num, dim_ffn, dim_embed, false);
  phi::DenseTensor ffn1_out;
  ffn1_out.Resize({token_num, dim_ffn});
  auto *ffn1_out_data =
      dev_ctx.template Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

  // 7. ffn act + bias
  DropoutParam ffn1_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
  FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
      dev_ctx, token_num, dim_ffn, ffn1_dropout_param);
  phi::DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
  ffn1_dropout_out.Resize({token_num, dim_ffn});
  auto *ffn1_dropout_out_data = dev_ctx.template Alloc<T>(
      &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));
  ffn1_dropout_mask.Resize({token_num, dim_ffn});
  auto *ffn1_dropout_mask_data = dev_ctx.template Alloc<uint8_t>(
      &ffn1_dropout_mask, ffn1_dropout_mask.numel() * sizeof(uint8_t));

  VLOG(1) << "5555555555555";
  // 8. ffn2 matmul
  auto ffn2_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, token_num, dim_embed, dim_ffn, false);

  // 9. ffn2 residual bias
  DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
  FusedDropoutLayerNormHelper<T, uint8_t> ffn2_fused_dropout_helper(
      dev_ctx, token_num, dim_embed, ffn2_dropout_param, epsilon);

  VLOG(1) << "aaaaaaaa";
  // calc
  auto *from_data = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  VLOG(1) << "auto *from_data = dev_ctx.template Alloc<T>(out, out->numel() * "
             "sizeof(T));";
  phi::DenseTensor *from_tensor = out;
  phi::DenseTensor tmp_out, tmp_out_rm_padding;
  tmp_out.Resize({token_num, dim_embed});
  VLOG(1) << "if (encoder_remove_padding)";
  if (encoder_remove_padding) {
    tmp_out_rm_padding.Resize({token_num, dim_embed});
    auto *tmp_out_rm_padding_data = dev_ctx.template Alloc<T>(
        &tmp_out_rm_padding, tmp_out_rm_padding.numel() * sizeof(T));
  }
  VLOG(1) << "auto *tmp_out_data";
  auto *tmp_out_data =
      dev_ctx.template Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

  VLOG(1) << "bbbbbbbbb";
  const T *x_data;
  if (encoder_remove_padding) {
    x_data = x_remove_padding.data<T>();
  } else {
    x_data = x.data<T>();
  }
  phi::DenseTensor *buf0 = nullptr;
  phi::DenseTensor *buf1 = nullptr;

  VLOG(1) << "ccccccc";
  // step0:  x   --> buf1
  // step1: buf1 --> buf0
  // step2: buf0 --> buf1
  int layers = qkv_weights.size();
  if (encoder_remove_padding) {
    // In the case of variable lengths, the padding needs to be rebuilt
    // eventually. So buf0 and buf1 do not need to be changed according to the
    // pre_layer_norm and the number of layers.
    buf0 = &tmp_out;
    buf1 = &tmp_out_rm_padding;
  } else {
    if (pre_layer_norm) {
      if (layers & 1) {
        // odd, set buf1 as out
        buf0 = &tmp_out;
        buf1 = out;
      } else {
        // even, set buf0 as out
        buf0 = out;
        buf1 = &tmp_out;
      }
    } else {
      buf0 = &tmp_out;
      buf1 = out;
    }
  }

  VLOG(1) << "ddddddddd";
  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    if (i == 0 && pre_layer_norm) {
      auto *ln_scale_data = ln_scales[i]->data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      // TODO(wangxi): can remove mean var in inference
      ln_compute.ComputeForward(x_data,
                                ln_scale_data,
                                ln_bias_data,
                                buf1->data<T>(),
                                ln_mean_data,
                                ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step1";
#endif

    VLOG(1) << "eeeeeeeeee";
    // step2. qkv
    const phi::DenseTensor *qkv_bias =
        qkv_biases && !qkv_biases.get().empty() ? qkv_biases.get()[i] : nullptr;
    // NOTE: in decoder stage, bias is fused in fmha
    const phi::DenseTensor *bias = time_step_t ? nullptr : qkv_bias;
    if (!pre_layer_norm && i == 0) {
      const phi::DenseTensor *tmp_input_x =
          (encoder_remove_padding) ? &x_remove_padding : &x;
      qkv_compute.ComputeForward(
          qkv_weights[i], tmp_input_x, bias, &qkv_out, &qkv_out);
    } else {
      qkv_compute.ComputeForward(
          qkv_weights[i], buf1, bias, &qkv_out, &qkv_out);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step2";
#endif

    VLOG(1) << "ffffffffff";
    // step3. fmha
    const phi::DenseTensor *cache_kv =
        cache_kvs && cache_kvs.get().size() > 0 ? cache_kvs.get()[i] : nullptr;
    phi::DenseTensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

    VLOG(1) << "6666666666666666";
    if (time_step_t) {  // generation decoder stage
      VLOG(1) << "aaaaaaaaa";
      // [2, batch_size, num_head, max_seq_len, head_size]
      int max_seq_len = cache_kv->dims()[3];
      VLOG(1) << "aaaaaaaaa";
      fmha<T>(dev_ctx,
              qkv_out,
              *qkv_bias,
              *src_mask_t,
              seq_lengths_t,
              rotary_tensor_t,
              cache_kv_out,
              &fmha_out,
              bsz,
              max_seq_len,
              num_head,
              dim_head,
              time_step_t->data<int>()[0],
              rotary_emb_dims,
              1. / std::sqrt(dim_head));
    } else if (cache_kv_out) {  // generation context stage
      VLOG(1) << "aaaaaaaaa";
      const phi::DenseTensor *pre_cache_kv_tensor =
          pre_caches && pre_caches.get().size() > 0 ? pre_caches.get()[i]
                                                    : nullptr;
      phi::DenseTensor *pre_cache_kv_out_tmp =
          cache_offset > 0 ? &pre_cache_kv_out : nullptr;
      phi::DenseTensor *src_mask_tmp =
          cache_offset > 0 ? &src_mask_out : nullptr;
      VLOG(1) << "bbbbbb";
      qkv_bias_add_transpose_split<T>(dev_ctx,
                                      q_transpose_out_data,
                                      kv_transpose_out_data,
                                      qkv_out_data,
                                      qkv_bias->data<T>(),
                                      padding_offset_data,
                                      token_num,
                                      bsz,
                                      num_head,
                                      seq_len,
                                      dim_head,
                                      compute_bias);

      VLOG(1) << "cccccccc";
      // q_transpose_out_data [bs, head_num, seq_len, dim_head]
      // kv_transpose_out_data [2, bs, head_num, seq_len, dim_head]
      if (rotary_emb_dims != 0) {
        auto *rotary_emb_data = rotary_tensor_t->data<T>();
        const int *sequence_lengths_data =
            encoder_remove_padding ? seq_lengths_t->data<int>() : nullptr;
        rotary_qk(dev_ctx,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  rotary_emb_data,
                  sequence_lengths_data,
                  rotary_emb_dims,
                  bsz,
                  num_head,
                  seq_len,
                  dim_head);
      }

      VLOG(1) << "ddddddddddd";
      phi::DenseTensor *tmp_padding_offset_tensor =
          encoder_remove_padding ? &padding_offset_tensor : nullptr;
      fmha_compute.ComputeForwardWithoutTranspose(pre_cache_kv_tensor,
                                                  src_mask_t,
                                                  tmp_padding_offset_tensor,
                                                  &q_transpose_out,
                                                  &kv_transpose_out,
                                                  pre_cache_kv_out_tmp,
                                                  &qk_out,
                                                  src_mask_tmp,
                                                  &softmax_out,
                                                  &attn_dropout_mask_out,
                                                  &attn_dropout_out,
                                                  &qktv_out,
                                                  &fmha_out,
                                                  token_num);
      const T *k_ptr = nullptr;
      const T *v_ptr = nullptr;
      VLOG(1) << "eeeeeeeeeeeee";
      if (cache_offset > 0) {
        // [2, bsz, num_head, cache_offset + seq_len, head_dim]
        const T *kv_data = pre_cache_kv_out.data<T>();
        k_ptr = kv_data;
        int64_t k_size = bsz * num_head * (seq_len + cache_offset) * dim_head;
        v_ptr = k_ptr + k_size;
      } else {
        // [3, bsz, num_head, seq_len, head_dim]
        int64_t k_size = bsz * seq_len * num_head * dim_head;
        const T *q_ptr = q_transpose_out_data;
        k_ptr = kv_transpose_out_data;
        v_ptr = k_ptr + k_size;
      }

      VLOG(1) << "fffffffff";
      // [2, bsz, num_head, max_seq_len, head_dim]
      int max_seq_len = cache_kv_out->dims()[3];
      T *cache_kv_data = cache_kv_out->data<T>();
      int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

      T *cache_k_ptr = cache_kv_data;
      T *cache_v_ptr = cache_kv_data + cache_k_size;
      VLOG(1) << "gggggggg";
      const int seq_len_tmp = seq_len + cache_offset;
      write_cache_kv<T>(dev_ctx,
                        cache_k_ptr,
                        cache_v_ptr,
                        k_ptr,
                        v_ptr,
                        bsz,
                        num_head,
                        seq_len_tmp,
                        max_seq_len,
                        dim_head);
    } else {  // not generation
      VLOG(1) << "ccccccccc";
      // TODO(wangxi): can remove dropout in inference
      qkv_bias_add_transpose_split<T>(dev_ctx,
                                      q_transpose_out_data,
                                      kv_transpose_out_data,
                                      qkv_out_data,
                                      qkv_bias->data<T>(),
                                      padding_offset_data,
                                      token_num,
                                      bsz,
                                      num_head,
                                      seq_len,
                                      dim_head,
                                      compute_bias);

      // q_transpose_out_data [bs, head_num, seq_len, dim_head]
      // kv_transpose_out_data [2, bs, head_num, seq_len, dim_head]
      if (rotary_emb_dims != 0) {
        auto *rotary_emb_data = rotary_tensor_t->data<T>();
        const int *sequence_lengths_data =
            encoder_remove_padding ? seq_lengths_t->data<int>() : nullptr;
        rotary_qk(dev_ctx,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  q_transpose_out_data,
                  kv_transpose_out_data,
                  rotary_emb_data,
                  sequence_lengths_data,
                  rotary_emb_dims,
                  bsz,
                  num_head,
                  seq_len,
                  dim_head);
      }

      phi::DenseTensor *tmp_padding_offset_tensor =
          encoder_remove_padding ? &padding_offset_tensor : nullptr;
      fmha_compute.ComputeForwardWithoutTranspose(cache_kv,
                                                  src_mask_t,
                                                  tmp_padding_offset_tensor,
                                                  &q_transpose_out,
                                                  &kv_transpose_out,
                                                  cache_kv_out,
                                                  &qk_out,
                                                  nullptr,
                                                  &softmax_out,
                                                  &attn_dropout_mask_out,
                                                  &attn_dropout_out,
                                                  &qktv_out,
                                                  &fmha_out,
                                                  token_num);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step3";
#endif

    if (pre_layer_norm) {
      out_linear_compute.ComputeForward(
          out_linear_weights[i], &fmha_out, nullptr, buf1, nullptr);
      AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
    } else {
      out_linear_compute.ComputeForward(
          out_linear_weights[i], &fmha_out, nullptr, buf0, nullptr);
      AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step4";
#endif

    VLOG(1) << "77777777777777";
    // step5. ln(residual + dropout(input + bias))
    if (pre_layer_norm) {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases.get()[i]->data<T>();

      // inplace
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf1->data<T>(),
          x_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    } else {
      auto *ln_scale_data = ln_scales[i]->data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases.get()[i]->data<T>();
      auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf0->data<T>(),
          residual_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step5";
#endif

    // step6. ffn matmul1
    ffn1_linear_compute.ComputeForward(
        ffn1_weights[i], buf1, nullptr, &ffn1_out, nullptr);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step6";
#endif

    // step7. act bias
    // TODO(wangxi): remove dropout mask in inference
    fused_act_dropout_helper.DropoutActBias(dev_ctx,
                                            ffn1_out_data,
                                            ffn1_biases.get()[i]->data<T>(),
                                            act_method,
                                            ffn1_dropout_out_data,
                                            ffn1_dropout_mask_data);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step7";
#endif

    // step8. ffn matmul2
    if (pre_layer_norm) {
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_dropout_out, nullptr, buf1, nullptr);
    } else {
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_dropout_out, nullptr, buf0, nullptr);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step8.0";
#endif

    if (pre_layer_norm) {
      AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
    } else {
      AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step8.1";
#endif

    VLOG(1) << "8888888888";
    // step9. residual bias
    if (pre_layer_norm) {
      // TODO(wangxi): remove dropout mask in inference
      if (i < layers - 1) {
        auto *ln_scale_data = ln_scales[i + 1]->data<U>();
        auto *ln_bias_data = ln_biases[i + 1]->data<U>();
        ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
            dev_ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            ffn2_biases.get()[i]->data<T>(),
            ln_scale_data,
            ln_bias_data,
            buf1->data<T>(),
            dropout_mask_out_data,
            buf0->data<T>(),
            ln_mean_data,
            ln_var_data);
      } else {
        ffn2_fused_dropout_helper.ResidualDropoutBias(
            dev_ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            ffn2_biases.get()[i]->data<T>(),
            buf1->data<T>(),
            dropout_mask_out_data);
      }
    } else {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
          dev_ctx,
          buf0->data<T>(),
          buf1->data<T>(),
          ffn2_biases.get()[i]->data<T>(),
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step9";
#endif
    if (pre_layer_norm) {
      x_data = buf1->data<T>();
      std::swap(buf0, buf1);
    }
  }
  if (encoder_remove_padding) {
    if (pre_layer_norm) {
      InvokeRebuildPadding(dev_ctx,
                           from_data,
                           buf0->data<T>(),
                           padding_offset_data,
                           token_num,
                           dim_embed);
    } else {
      InvokeRebuildPadding(dev_ctx,
                           from_data,
                           buf1->data<T>(),
                           padding_offset_data,
                           token_num,
                           dim_embed);
    }
  }
}
#endif  // CUDA_VERSION >= 11060

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multi_transformer,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(8).SetBackend(phi::Backend::CPU);
}
