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

/*
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle_analysis_config.h"  // NOLINT
#include "paddle_api.h"              // NOLINT

///
/// \file paddle_inference_api.h
///
/// \brief Paddle Inference API
///
/// \author paddle-infer@baidu.com
/// \date 2020-09-01
/// \since 2.0.0-beta
///

namespace paddle_infer {

using PrecisionType = paddle::AnalysisConfig::Precision;
using Config = paddle::AnalysisConfig;
using DistConfig = paddle::DistConfig;

///
/// \class Predictor
///
/// \brief Predictor is the interface for model prediction.
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output handle
/// \code{cpp}
///   auto input_t = predictor->GetInputHandle(input_names[0]);
///   auto output_t = predictor->GetOutputHandle(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->Run();
/// \endcode
///
class PD_INFER_DECL Predictor {
 public:
  Predictor() = delete;
  ~Predictor() {}
  // Use for clone
  explicit Predictor(std::unique_ptr<paddle::PaddlePredictor>&& pred)
      : predictor_(std::move(pred)) {}

  ///
  /// \brief Construct a new Predictor object
  ///
  /// \param[in] Config config
  ///
  explicit Predictor(const Config& config);

  ///
  /// \brief Get all input names and their corresponding type
  ///
  /// \return the map of input names and type
  ///
  std::map<std::string, DataType> GetInputTypes();

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames();

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  std::unique_ptr<Tensor> GetInputHandle(const std::string& name);

  ///
  /// \brief Run the prediction engine
  ///
  /// \return Whether the function executed successfully
  ///
  bool Run();

  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  std::vector<std::string> GetOutputNames();

  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  std::unique_ptr<Tensor> GetOutputHandle(const std::string& name);

  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  std::unique_ptr<Predictor> Clone(void* stream = nullptr);

  /// \brief Clear the intermediate tensors of the predictor
  void ClearIntermediateTensor();

  ///
  /// \brief Release all tmp tensor to compress the size of the memory pool.
  /// The memory pool is considered to be composed of a list of chunks, if
  /// the chunk is not occupied, it can be released.
  ///
  /// \return Number of bytes released. It may be smaller than the actual
  /// released memory, because part of the memory is not managed by the
  /// MemoryPool.
  ///
  uint64_t TryShrinkMemory();

  ///
  /// \brief Register a output hook function to operate the intermediate tensor
  /// of op output. when using this function, memory reuse should be tured off.
  /// The hook function signature is void(const std::string&, const
  /// std::string&, const Tensor&>). Here, the first parameter is op's
  /// type, the second param is output var name of the op, and the third
  /// parameter is output tensor with the var name.
  ///
  void RegisterOutputHook(const Exp_OutputHookFunc& hookfunc);

  ///
  /// \brief Get the execution stream on devices with a concept of stream,
  /// otherwise returns nullptr.
  ///
  /// \return The execution stream or nullptr (CPU).
  ///
  void* GetExecStream() const;

 private:
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  friend class paddle_infer::experimental::InternalUtils;
};

///
/// \brief A factory to help create predictors.
///
/// Usage:
///
/// \code{.cpp}
/// Config config;
/// ... // change the configs.
/// auto predictor = CreatePredictor(config);
/// \endcode
///
PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    const Config& config);  // NOLINT

PD_INFER_DECL int GetNumBytesOfDataType(DataType dtype);

PD_INFER_DECL std::string GetVersion();
PD_INFER_DECL std::tuple<int, int, int> GetTrtCompileVersion();
PD_INFER_DECL std::tuple<int, int, int> GetTrtRuntimeVersion();
PD_INFER_DECL std::string UpdateDllFlag(const char* name, const char* value);

PD_INFER_DECL void ConvertToMixedPrecision(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& mixed_model_file,
    const std::string& mixed_params_file,
    PrecisionType mixed_precision,
    PlaceType backend,
    bool keep_io_types = true,
    std::unordered_set<std::string> black_list = {});

enum DistModelDataType {
  DIST_FLOAT16,
  DIST_FLOAT32,
  DIST_INT64,
  DIST_INT32,
  DIST_INT8
};

class PD_INFER_DECL DistModelDataBuf {
 public:
  DistModelDataBuf() = default;
  void Reshape(const std::vector<int>& shape) {
    shape_ = shape;
    int size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    len_ = static_cast<size_t>(size);
  }
  template <typename T>
  void CopyFromCpu(T* data);

  template <typename T>
  T* GetData() {
    return reinterpret_cast<T*>(data_.data());
  }

  char* data() { return data_.data(); }
  DistModelDataType dtype() { return dtype_; }
  void SetDtype(int dtype) { dtype_ = static_cast<DistModelDataType>(dtype); }
  size_t size() { return len_; }
  std::vector<int> shape() { return shape_; }

  std::vector<char> data_;
  DistModelDataType dtype_;

 private:
  size_t len_;
  std::vector<int> shape_;
};

struct PD_INFER_DECL DistModelPredictorConfig {
  std::string model_dir;
  std::string place;
};

class PD_INFER_DECL DistModelPredictorBase {
 public:
  DistModelPredictorBase() = default;
  DistModelPredictorBase(const DistModelPredictorBase&) = delete;
  DistModelPredictorBase& operator=(const DistModelPredictorBase&) = delete;

  virtual bool Run() = 0;
  virtual std::vector<std::string> GetInputNames() = 0;
  virtual void SetInput(const std::string& name,
                        DistModelDataBuf* data_buf,
                        std::vector<std::vector<size_t>> lod) = 0;
  virtual std::vector<std::string> GetOutputNames() = 0;
  virtual std::vector<int> GetOutputShape(const std::string& name) = 0;
  virtual DistModelDataBuf GetOutputData(const std::string& name) = 0;
};

PD_INFER_DECL std::shared_ptr<DistModelPredictorBase> CreateDistModelPredictor(
    const DistModelPredictorConfig& config);

namespace services {
///
/// \class PredictorPool
///
/// \brief PredictorPool is a simple encapsulation of Predictor, suitable for
/// use in multi-threaded situations. According to the thread id, the
/// corresponding Predictor is taken out from PredictorPool to complete the
/// prediction.
///
class PD_INFER_DECL PredictorPool {
 public:
  PredictorPool() = delete;
  PredictorPool(const PredictorPool&) = delete;
  PredictorPool& operator=(const PredictorPool&) = delete;

  /// \brief Construct the predictor pool with \param size predictor instances.
  explicit PredictorPool(const Config& config, size_t size = 1);

  /// \brief Get \param id-th predictor.
  Predictor* Retrive(size_t idx);

 private:
  std::shared_ptr<Predictor> main_pred_;
  std::vector<std::unique_ptr<Predictor>> preds_;
};
}  // namespace services

}  // namespace paddle_infer
