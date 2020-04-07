# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, Layer, to_variable
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
from model import Model, CrossEntropy, Loss
from text import TransformerBeamSearchDecoder, DynamicDecode


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


class NoamDecay(LearningRateDecay):
    """
    learning rate scheduler
    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 static_lr=2.0,
                 begin=1,
                 step=1,
                 dtype='float32'):
        super(NoamDecay, self).__init__(begin, step, dtype)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.static_lr = static_lr

    def step(self):
        a = self.create_lr_var(self.step_num**-0.5)
        b = self.create_lr_var((self.warmup_steps**-1.5) * self.step_num)
        lr_value = (self.d_model**-0.5) * layers.elementwise_min(
            a, b) * self.static_lr
        return lr_value


class PrePostProcessLayer(Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y else x)
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" % len(
                            self.sublayers(include_sublayers=False)),
                        LayerNorm(
                            normalized_shape=d_model,
                            param_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(1.)),
                            bias_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(0.)))))
            elif cmd == "d":  # add dropout
                self.functors.append(lambda x: layers.dropout(
                    x, dropout_prob=dropout_rate, is_test=False)
                                     if dropout_rate else x)

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class MultiHeadAttention(Layer):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = Linear(
            input_dim=d_model, output_dim=d_key * n_head, bias_attr=False)
        self.k_fc = Linear(
            input_dim=d_model, output_dim=d_key * n_head, bias_attr=False)
        self.v_fc = Linear(
            input_dim=d_model, output_dim=d_value * n_head, bias_attr=False)
        self.proj_fc = Linear(
            input_dim=d_value * n_head, output_dim=d_model, bias_attr=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:  # self-attention
            keys, values = queries, queries
            static_kv = False
        else:  # cross-attention
            static_kv = True

        q = self.q_fc(queries)
        q = layers.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = layers.transpose(x=q, perm=[0, 2, 1, 3])

        if cache is not None and static_kv and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = layers.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
            k = layers.transpose(x=k, perm=[0, 2, 1, 3])
            v = layers.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
            v = layers.transpose(x=v, perm=[0, 2, 1, 3])

        if cache is not None:
            if static_kv and not "static_k" in cache:
                # for encoder-decoder attention in inference and has not cached
                cache["static_k"], cache["static_v"] = k, v
            elif not static_kv:
                # for decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]
                k = layers.concat([cache_k, k], axis=2)
                v = layers.concat([cache_v, v], axis=2)
                cache["k"], cache["v"] = k, v

        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(queries, keys, values, cache)

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.d_model**-0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if self.dropout_rate:
            weights = layers.dropout(
                weights, dropout_prob=self.dropout_rate, is_test=False)

        out = layers.matmul(weights, v)

        # combine heads
        out = layers.transpose(out, perm=[0, 2, 1, 3])
        out = layers.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.proj_fc(out)
        return out

    def cal_kv(self, keys, values):
        k = self.k_fc(keys)
        v = self.v_fc(values)
        k = layers.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
        k = layers.transpose(x=k, perm=[0, 2, 1, 3])
        v = layers.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
        v = layers.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v


class FFN(Layer):
    """
    Feed-Forward Network
    """

    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = Linear(
            input_dim=d_model, output_dim=d_inner_hid, act="relu")
        self.fc2 = Linear(input_dim=d_inner_hid, output_dim=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.dropout_rate:
            hidden = layers.dropout(
                hidden, dropout_prob=self.dropout_rate, is_test=False)
        out = self.fc2(hidden)
        return out


class EncoderLayer(Layer):
    """
    EncoderLayer
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(
            self.preprocesser1(enc_input), None, None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)

        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class Encoder(Layer):
    """
    encoder
    """

    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(Encoder, self).__init__()

        self.encoder_layers = list()
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                                 prepostprocess_dropout, attention_dropout,
                                 relu_dropout, preprocess_cmd,
                                 postprocess_cmd)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output

        return self.processer(enc_output)


class Embedder(Layer):
    """
    Word Embedding + Position Encoding
    """

    def __init__(self, vocab_size, emb_dim, bos_idx=0):
        super(Embedder, self).__init__()

        self.word_embedder = Embedding(
            size=[vocab_size, emb_dim],
            padding_idx=bos_idx,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Normal(0., emb_dim**-0.5)))

    def forward(self, word):
        word_emb = self.word_embedder(word)
        return word_emb


class WrapEncoder(Layer):
    """
    embedder + encoder
    """

    def __init__(self, src_vocab_size, max_length, n_layer, n_head, d_key,
                 d_value, d_model, d_inner_hid, prepostprocess_dropout,
                 attention_dropout, relu_dropout, preprocess_cmd,
                 postprocess_cmd, word_embedder):
        super(WrapEncoder, self).__init__()

        self.emb_dropout = prepostprocess_dropout
        self.emb_dim = d_model
        self.word_embedder = word_embedder
        self.pos_encoder = Embedding(
            size=[max_length, self.emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    position_encoding_init(max_length, self.emb_dim)),
                trainable=False))

        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, preprocess_cmd,
                               postprocess_cmd)

    def forward(self, src_word, src_pos, src_slf_attn_bias):
        word_emb = self.word_embedder(src_word)
        word_emb = layers.scale(x=word_emb, scale=self.emb_dim**0.5)
        pos_enc = self.pos_encoder(src_pos)
        pos_enc.stop_gradient = True
        emb = word_emb + pos_enc
        enc_input = layers.dropout(
            emb, dropout_prob=self.emb_dropout,
            is_test=False) if self.emb_dropout else emb

        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class DecoderLayer(Layer):
    """
    decoder
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):
        super(DecoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.cross_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                             attention_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser3 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser3 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                cache=None):
        self_attn_output = self.self_attn(
            self.preprocesser1(dec_input), None, None, self_attn_bias, cache)
        self_attn_output = self.postprocesser1(self_attn_output, dec_input)

        cross_attn_output = self.cross_attn(
            self.preprocesser2(self_attn_output), enc_output, enc_output,
            cross_attn_bias, cache)
        cross_attn_output = self.postprocesser2(cross_attn_output,
                                                self_attn_output)

        ffn_output = self.ffn(self.preprocesser3(cross_attn_output))
        ffn_output = self.postprocesser3(ffn_output, cross_attn_output)

        return ffn_output


class Decoder(Layer):
    """
    decoder
    """

    def __init__(self, n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd):
        super(Decoder, self).__init__()

        self.decoder_layers = list()
        for i in range(n_layer):
            self.decoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    DecoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                                 prepostprocess_dropout, attention_dropout,
                                 relu_dropout, preprocess_cmd,
                                 postprocess_cmd)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                caches=None):
        for i, decoder_layer in enumerate(self.decoder_layers):
            dec_output = decoder_layer(dec_input, enc_output, self_attn_bias,
                                       cross_attn_bias, None
                                       if caches is None else caches[i])
            dec_input = dec_output

        return self.processer(dec_output)

    def prepare_static_cache(self, enc_output):
        return [
            dict(
                zip(("static_k", "static_v"),
                    decoder_layer.cross_attn.cal_kv(enc_output, enc_output)))
            for decoder_layer in self.decoder_layers
        ]


class WrapDecoder(Layer):
    """
    embedder + decoder
    """

    def __init__(self, trg_vocab_size, max_length, n_layer, n_head, d_key,
                 d_value, d_model, d_inner_hid, prepostprocess_dropout,
                 attention_dropout, relu_dropout, preprocess_cmd,
                 postprocess_cmd, share_input_output_embed, word_embedder):
        super(WrapDecoder, self).__init__()

        self.emb_dropout = prepostprocess_dropout
        self.emb_dim = d_model
        self.word_embedder = word_embedder
        self.pos_encoder = Embedding(
            size=[max_length, self.emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    position_encoding_init(max_length, self.emb_dim)),
                trainable=False))

        self.decoder = Decoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, preprocess_cmd,
                               postprocess_cmd)

        if share_input_output_embed:
            self.linear = lambda x: layers.matmul(x=x,
                                                  y=self.word_embedder.
                                                  word_embedder.weight,
                                                  transpose_y=True)
        else:
            self.linear = Linear(
                input_dim=d_model, output_dim=trg_vocab_size, bias_attr=False)

    def forward(self,
                trg_word,
                trg_pos,
                trg_slf_attn_bias,
                trg_src_attn_bias,
                enc_output,
                caches=None):
        word_emb = self.word_embedder(trg_word)
        word_emb = layers.scale(x=word_emb, scale=self.emb_dim**0.5)
        pos_enc = self.pos_encoder(trg_pos)
        pos_enc.stop_gradient = True
        emb = word_emb + pos_enc
        dec_input = layers.dropout(
            emb, dropout_prob=self.emb_dropout,
            is_test=False) if self.emb_dropout else emb
        dec_output = self.decoder(dec_input, enc_output, trg_slf_attn_bias,
                                  trg_src_attn_bias, caches)
        dec_output = layers.reshape(
            dec_output,
            shape=[-1, dec_output.shape[-1]], )
        logits = self.linear(dec_output)
        return logits


class CrossEntropyCriterion(Loss):
    def __init__(self, label_smooth_eps):
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps

    def forward(self, outputs, labels):
        predict, (label, weights) = outputs[0], labels
        if self.label_smooth_eps:
            label = layers.label_smooth(
                label=layers.one_hot(
                    input=label, depth=predict.shape[-1]),
                epsilon=self.label_smooth_eps)

        cost = layers.softmax_with_cross_entropy(
            logits=predict,
            label=label,
            soft_label=True if self.label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = layers.reduce_sum(weighted_cost)
        token_num = layers.reduce_sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return avg_cost


class Transformer(Model):
    """
    model
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1):
        super(Transformer, self).__init__()
        src_word_embedder = Embedder(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_idx=bos_id)
        self.encoder = WrapEncoder(
            src_vocab_size, max_length, n_layer, n_head, d_key, d_value,
            d_model, d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd, src_word_embedder)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            trg_word_embedder = src_word_embedder
        else:
            trg_word_embedder = Embedder(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_idx=bos_id)
        self.decoder = WrapDecoder(
            trg_vocab_size, max_length, n_layer, n_head, d_key, d_value,
            d_model, d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing,
            trg_word_embedder)

        self.trg_vocab_size = trg_vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value

    def forward(self, src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                trg_slf_attn_bias, trg_src_attn_bias):
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        predict = self.decoder(trg_word, trg_pos, trg_slf_attn_bias,
                               trg_src_attn_bias, enc_output)
        return predict


class TransfomerCell(object):
    """
    Let inputs=(trg_word, trg_pos), states=cache to make Transformer can be
    used as RNNCell
    """

    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, inputs, states, trg_src_attn_bias, enc_output,
                 static_caches):
        trg_word, trg_pos = inputs
        for cache, static_cache in zip(states, static_caches):
            cache.update(static_cache)
        logits = self.decoder(trg_word, trg_pos, None, trg_src_attn_bias,
                              enc_output, states)
        new_states = [{"k": cache["k"], "v": cache["v"]} for cache in states]
        return logits, new_states


class InferTransformer(Transformer):
    """
    model for prediction
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)  # py3
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        super(InferTransformer, self).__init__(**args)
        cell = TransfomerCell(self.decoder)
        self.beam_search_decoder = DynamicDecode(
            TransformerBeamSearchDecoder(
                cell, bos_id, eos_id, beam_size, var_dim_in_state=2),
            max_out_len,
            is_test=True)

    def forward(self, src_word, src_pos, src_slf_attn_bias, trg_src_attn_bias):
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = [{
            "k": layers.fill_constant_batch_size_like(
                input=enc_output,
                shape=[-1, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v": layers.fill_constant_batch_size_like(
                input=enc_output,
                shape=[-1, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]
        enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
            enc_output, self.beam_size)
        trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
            trg_src_attn_bias, self.beam_size)
        static_caches = self.decoder.decoder.prepare_static_cache(enc_output)
        rs, _ = self.beam_search_decoder(
            inits=caches,
            enc_output=enc_output,
            trg_src_attn_bias=trg_src_attn_bias,
            static_caches=static_caches)
        return rs
