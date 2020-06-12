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
class SkipLayerNormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit SkipLayerNormPluginDynamic(const float* bias, const float* scale,
                                      int bias_size, int scale_size, float eps,
                                      bool ban_fp16)
      : bias_size_(bias_size),
        scale_size_(scale_size),
        eps_(eps),
        ban_fp16_(ban_fp16) {
    bias_.resize(bias_size);
    scale_.resize(scale_size);
    std::copy(bias, bias + bias_size, bias_.data());
    std::copy(scale, scale + scale_size, scale_.data());
  }
  SkipLayerNormPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_size_);
    DeserializeValue(&serialData, &serialLength, &scale_size_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &ban_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new SkipLayerNormPluginDynamic(
        bias_.data(), scale_.data(), bias_size_, scale_size_, eps_, ban_fp16_);
  }

  const char* getPluginType() const override { return "skip_layernorm_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override {
    size_t ser_size = SerializedSize(bias_) + SerializedSize(scale_) +
                      SerializedSize(bias_size_) + SerializedSize(scale_size_) +
                      SerializedSize(eps_) + SerializedSize(eps_);
    return ser_size;
  }
  void serialize(void* buffer) const override {
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, ban_fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void destroy() override { delete this; }

 private:
  std::vector<float> bias_;
  std::vector<float> scale_;

  float* bias_gpu_;
  float* scale_gpu_;

  int bias_size_;
  int scale_size_;

  float eps_;
  bool ban_fp16_;
};

class SkipLayerNormPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  SkipLayerNormPluginV2Creator() {}
  const char* getPluginName() const override { return "skip_layernorm_plugin"; }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFieldCollection;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return new SkipLayerNormPluginDynamic();
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    auto plugin = new SkipLayerNormPluginDynamic(serialData, serialLength);
    return plugin;
  }

  void setPluginNamespace(const char* libNamespace) override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  std::string mNamespace;
  std::string mPluginName;
  nvinfer1::PluginFieldCollection mFieldCollection;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};
REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginV2Creator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
