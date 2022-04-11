// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
class PrelnResidualBiasPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit PrelnResidualBiasPluginDynamic(const float* bias, const float* scale, const float* ele_bias,
                                      int bias_size, int scale_size, int ele_bias_size,
                                      const float eps, bool with_fp16)
      : bias_size_(bias_size), scale_size_(scale_size), ele_bias_size_(ele_bias_size), eps_(eps) {
    with_fp16_ = with_fp16;
    bias_.resize(bias_size);
    scale_.resize(scale_size);
    ele_bias_.resize(ele_bias_size);
    std::copy(bias, bias + bias_size, bias_.data());
    std::copy(scale, scale + scale_size, scale_.data());
    std::copy(ele_bias, ele_bias + ele_bias_size, ele_bias_.data());
  }

  PrelnResidualBiasPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &bias_);
    DeserializeValue(&serial_data, &serial_length, &scale_);
    DeserializeValue(&serial_data, &serial_length, &ele_bias_);
    DeserializeValue(&serial_data, &serial_length, &bias_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_size_);
    DeserializeValue(&serial_data, &serial_length, &ele_bias_size_);
    DeserializeValue(&serial_data, &serial_length, &eps_);
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
  }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto ptr = new PrelnResidualBiasPluginDynamic(
        bias_.data(), scale_.data(), ele_bias_.data(), bias_size_, scale_size_, ele_bias_size_, eps_, with_fp16_);
    ptr->bias_gpu_ = bias_gpu_;
    ptr->scale_gpu_ = scale_gpu_;
    ptr->ele_bias_gpu_ = ele_bias_gpu_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "skip_layernorm_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 2; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    size_t ser_size = SerializedSize(bias_) + SerializedSize(scale_) + SerializedSize(ele_bias_) +
                      SerializedSize(bias_size_) + SerializedSize(scale_size_) + SerializedSize(ele_bias_size_) +
                      SerializedSize(eps_) + SerializedSize(with_fp16_);
    return ser_size;
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, ele_bias_);
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    SerializeValue(&buffer, ele_bias_size_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, with_fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
  void terminate() TRT_NOEXCEPT override;

 private:
  std::vector<float> bias_;
  std::vector<float> scale_;
  std::vector<float> ele_bias_;

  float* bias_gpu_{nullptr};
  float* scale_gpu_{nullptr};
  float* ele_bias_gpu_{nullptr};

  int bias_size_;
  int scale_size_;
  int ele_bias_size_;

  float eps_;
};

class PrelnResidualBiasPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  PrelnResidualBiasPluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "preln_residual_bias_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    auto plugin = new PrelnResidualBiasPluginDynamic(serial_data, serial_length);
    return plugin;
  }

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(PrelnResidualBiasPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
