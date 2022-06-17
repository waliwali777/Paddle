/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/elementwise_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class ElementwiseTensorOpConverter : public OpConverter {
 public:
  ElementwiseTensorOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "Convert a fluid elementwise op to TensorRT IElementWiseLayer";
    framework::OpDesc op_desc(op, nullptr);
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::ITensor* Y = nullptr;
    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    if (Y_v) {
      // Y is weight
      auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
      float* weight_data =
          engine_->GetWeightCPUData(op_desc.Input("Y").front(), Y_t);
      std::vector<int> dims_y = phi::vectorize<int>(Y_t->dims());
      TensorRTEngine::Weight y_weight{nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(weight_data),
                                      static_cast<size_t>(Y_t->numel())};
      nvinfer1::Dims trt_dims_y;
      trt_dims_y.nbDims = dims_y.size();
      for (int i = 0; i < trt_dims_y.nbDims; i++) {
        trt_dims_y.d[i] = dims_y[i];
      }
      Y = TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_dims_y, y_weight.get())
              ->getOutput(0);
    } else {
      Y = engine_->GetITensor(op_desc.Input("Y").front());
    }

    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();
    auto output_name = op_desc.Output("Out")[0];

    // axis here is relative to explicit batch
    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));
    int real_x_rank = dims_x.nbDims;
    int real_y_rank = dims_y.nbDims;
    if (!engine_->with_dynamic_shape()) {
      real_x_rank++;
      real_y_rank++;
      if (Y_v) real_y_rank--;
    }
    if (axis == -1) {
      axis = real_x_rank - real_y_rank;
    }
    if (!engine_->with_dynamic_shape() && axis > 0) {
      axis--;
    }

    // X: - -  -    - - - -
    //        axis
    // Y:      -    - -
    // we need expand Y's rank = X's rank
    int left_one_num = axis;
    int right_one_num = dims_x.nbDims - axis - dims_y.nbDims;
    nvinfer1::IShuffleLayer* reshape_layer;
    if (engine_->with_dynamic_shape()) {
      auto* y_shape_tensor = Shape(Y);
      auto* new_y_shape_tensor = y_shape_tensor;
      if (axis > 0) {
        std::vector<int32_t> left_one(left_one_num, 1);
        auto* left_one_tensor = Add1DConstantLayer(left_one);
        new_y_shape_tensor = Concat(std::vector<nvinfer1::ITensor*>{
            left_one_tensor, new_y_shape_tensor});
      }
      if (right_one_num > 0) {
        std::vector<int32_t> right_one(right_one_num, 1);
        auto* right_one_tensor = Add1DConstantLayer(right_one);
        new_y_shape_tensor = Concat(std::vector<nvinfer1::ITensor*>{
            new_y_shape_tensor, right_one_tensor});
      }
      reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *Y);
      reshape_layer->setInput(1, *new_y_shape_tensor);
    } else {
      nvinfer1::Dims new_y_dims;
      new_y_dims.nbDims = left_one_num + dims_y.nbDims + right_one_num;
      for (int i = 0; i < new_y_dims.nbDims; i++) new_y_dims.d[i] = 1;
      for (int i = 0; i < dims_y.nbDims; i++)
        new_y_dims.d[left_one_num + i] = dims_y.d[i];
      reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *Y);
      reshape_layer->setReshapeDimensions(new_y_dims);
    }

    auto* reshape_y_tensor = reshape_layer->getOutput(0);

    auto op_pair = ops.find(op_type_);
    PADDLE_ENFORCE_NE(op_pair, ops.end(),
                      platform::errors::InvalidArgument(
                          "Elementwise op's type(%s) is not supported. Please "
                          "check if the op_type is correct.",
                          op_type_));

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *X,
                                       *reshape_y_tensor, op_pair->second);
    RreplenishLayerAndOutput(layer, "elementwise", {output_name}, test_mode);
  }

 protected:
  static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
      ops;
  std::string op_type_;
};

const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
    ElementwiseTensorOpConverter::ops = {
        {"add", nvinfer1::ElementWiseOperation::kSUM},
        {"mul", nvinfer1::ElementWiseOperation::kPROD},
        {"sub", nvinfer1::ElementWiseOperation::kSUB},
        {"div", nvinfer1::ElementWiseOperation::kDIV},
        {"min", nvinfer1::ElementWiseOperation::kMIN},
        {"pow", nvinfer1::ElementWiseOperation::kPOW},
        {"max", nvinfer1::ElementWiseOperation::kMAX},
};

class ElementwiseTensorAddOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorAddOpConverter() { op_type_ = "add"; }
};

class ElementwiseTensorMulOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMulOpConverter() { op_type_ = "mul"; }
};

class ElementwiseTensorSubOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorSubOpConverter() { op_type_ = "sub"; }
};

class ElementwiseTensorDivOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorDivOpConverter() { op_type_ = "div"; }
};

class ElementwiseTensorMinOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMinOpConverter() { op_type_ = "min"; }
};

class ElementwiseTensorMaxOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMaxOpConverter() { op_type_ = "max"; }
};

class ElementwiseTensorPowOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorPowOpConverter() { op_type_ = "pow"; }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(elementwise_add_weight,
                          ElementwiseTensorAddOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_mul_weight,
                          ElementwiseTensorMulOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_sub_weight,
                          ElementwiseTensorSubOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_div_weight,
                          ElementwiseTensorDivOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_pow_weight,
                          ElementwiseTensorPowOpConverter);

REGISTER_TRT_OP_CONVERTER(elementwise_add_tensor,
                          ElementwiseTensorAddOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_sub_tensor,
                          ElementwiseTensorSubOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_div_tensor,
                          ElementwiseTensorDivOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_mul_tensor,
                          ElementwiseTensorMulOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_max_tensor,
                          ElementwiseTensorMaxOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_min_tensor,
                          ElementwiseTensorMinOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_pow_tensor,
                          ElementwiseTensorPowOpConverter);
