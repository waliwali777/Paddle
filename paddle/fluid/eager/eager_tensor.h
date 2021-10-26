// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <Python.h>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/pten/hapi/all.h"

/**
 * This class is used by Eager mode for now. It's painful to do this in Eager
 * Mode, the better
 * choice is to use paddle::experimental::Tensor directly. However, we have a
 * punch of nested kernel code, and
 * they use paddle::framework::Variable in inner logic code. So, we have to
 * provide variable in
 * paddle::framework::ExecutionContext to support it. We should remove this as
 * soon as we finish our latest
 * Pten Lib, and use paddle::experimental::Tensor instead.
 *
 * Note: Keep this class as clean as possible.
 * This class should only support method declared in
 * paddle::experimental::Tensor with access method of
 * paddle::framework::Variable no more members are acceptable.
 * **/

namespace egr {
class EagerTensor final {
 public:
  /* Part 1: Constructors */
  EagerTensor()
      : tensor_(std::make_shared<paddle::experimental::Tensor>()),
        var_(paddle::framework::Variable()) {}
  explicit EagerTensor(const std::string& name)
      : tensor_(std::make_shared<paddle::experimental::Tensor>(name)),
        var_(paddle::framework::Variable()) {}
  /**
   * @description: Use a TensorImpl pointer to construct a Tensor
   * @param {shared_ptr<TensorBase>} tensor_impl
   * @return {Tensor}
   */
  explicit EagerTensor(const std::shared_ptr<pten::TensorBase>& tensor_impl)
      : tensor_(std::make_shared<paddle::experimental::Tensor>(tensor_impl)),
        var_(paddle::framework::Variable()) {}
  EagerTensor(const EagerTensor&) = default;
  EagerTensor(EagerTensor&&) = default;

  /* Part 2: Name access methods */
  /**
   * @description: Return the name of current Tensor.
   * @param None
   * @return {const std::string&}
   */
  const std::string& name() const { return tensor_->name(); }
  /**
   * @description: Set the name of current Tensor.
   * @param {const std::string& name}
   * @return None
   */
  void set_name(const std::string& name) { tensor_->set_name(name); }

  /* Part 3: Dimension, DataType and DataLayout methods */
  /**
   * @description: Return the number of elements of current Tensor.
   * @param None
   * @return {int64_t}
   */
  int64_t numel() const { return tensor_->impl()->numel(); }
  /**
   * @description: Return the shape (dimensions) of current Tensor.
   * @param None
   * @return {DDim}
   */
  paddle::framework::DDim shape() const { return tensor_->impl()->dims(); }

  /**
   * @description: Return the data type of current Tensor.
   * @param None
   * @return {DataType}
   */
  paddle::experimental::DataType type() const {
    return tensor_->impl()->data_type();
  }

  /**
   * @description: Return the layout of current Tensor.
   * @param None
   * @return {DataLayout}
   */
  paddle::experimental::DataLayout layout() const {
    return tensor_->impl()->layout();
  }

  /* Part 3: Device and Backend methods */
  /**
   * @description: Return the place (device) of current Tensor.
   * @param None
   * @return {Place}
   */
  paddle::platform::Place place() const { return tensor_->impl()->place(); }

  /**
   * Backend judgment APIs, shield the concept of Backend.
   */
  bool is_cpu() const { return paddle::platform::is_cpu_place(place()); }
  bool is_cuda() const { return paddle::platform::is_gpu_place(place()); }

  /* Part 4: Data Access methods */
  /**
   * @description: Return the implemention of current Tensor.
   * @param None
   * @return {std::shared_ptr<TensorBase>}
   */
  std::shared_ptr<pten::TensorBase> impl() const { return tensor_->impl(); }

  /**
   * @description: Set the implemention of current Tensor.
   * @param {std::shared_ptr<TensorBase>}
   * @return None
   */
  void set_impl(const std::shared_ptr<pten::TensorBase>& impl) {
    tensor_->set_impl(impl);
  }

  // TODO(chenweihang): Whether API Tensor need `data` and `mutable_data`?

  // TODO(chenweihang): slice and split methods use kernels?

  /* Part 5: Status utils methods */
  /**
   * @description: Determine whether it is a meaningful Tensor
   * @param None
   * @return {bool}
   */
  bool defined() const { return tensor_->impl() != nullptr; }

  /**
   * @description: Determine whether Tensor is initialized
   * @param None
   * @return {bool}
   */
  bool initialized() const { return tensor_->impl()->initialized(); }

  /**
   * @description: Reset the Tensor implementation
   * @param None
   * @return {void}
   */
  void reset() { tensor_->reset(); }

  /* Part 6: Operator overloading */
  EagerTensor& operator=(const EagerTensor& x) & {
    tensor_ = x.tensor_;
    var_ = x.var_;
    return *this;
  }
  EagerTensor& operator=(EagerTensor&& x) & {
    tensor_ = std::move(x.tensor_);
    var_ = std::move(x.var_);
    return *this;
  }

  /* Part 7: Autograd methods */
  paddle::experimental::AbstractAutogradMeta* get_autograd_meta() const {
    return tensor_->get_autograd_meta();
  }
  void set_autograd_meta(
      std::shared_ptr<paddle::experimental::AbstractAutogradMeta>
          autograd_meta) {
    tensor_->set_autograd_meta(autograd_meta);
  }

  /** Part 8: This should only be used in eager.cc, remove this when we support
   * contructor calling in python_C **/
  void set_tensor(const std::shared_ptr<paddle::experimental::Tensor>& tensor) {
    tensor_ = tensor;
  }

  /** Part 9: Get framework::Variable from EagerTensor **/
  const paddle::framework::Variable& Var() const { var_; }

  paddle::framework::Variable* MutableVar() { return &var_; }

  /** Part 10: Sync paddle::framework::Variable with pten::Tensor **/
  void SyncToVar(const paddle::framework::proto::VarType& type) {
    // Synchronize allocation only once.
    if (!var_.IsInitialized()) {
      // TODO(jiabin): Support selected rows later.
      PADDLE_ENFORCE(
          this->initialized(),
          paddle::platform::errors::Fatal("Can not Sync EagerTensor %s whose "
                                          "pten::Tensor is not initialized!" %
                                          name()));
      if ((paddle::framework::proto::VarType::LOD_TENSOR == type) ||
          (paddle::framework::proto::VarType::TENSOR == type)) {
        auto* framework_tensor =
            var_.GetMutable<paddle::framework::LoDTensor>();
        framework_tensor->Resize(tensor_->shape());
        framework_tensor->set_layout(
            pten::TransToFluidDataLayout(tensor.layout()));
        // Contruct framework::Tensor from egr::EagerTensor
        if (auto tensor_dense =
                std::dynamic_pointer_cast<pten::DenseTensor>(tensor_->impl())) {
          paddle::framework::ShareTensorImpl<pten::DenseTensor>(
              tensor_dense.get(), framework_tensor);

        } else {
          PADDLE_THROW(paddle::platform::errors::Fatal(
              "Unrecognized egr::EagerTensor type, only "
              "DenseTensor is supported for now."));
        }
      }
    }
  }

 private:
  template <typename LEGACY_TYPE, typename TYPE>
  void SetImplWithLegacyTensor() {
    pten::TensorMeta meta;
    const auto& framework_tensor = var_.Get<LEGACY_TYPE>();
    meta.dims = framework_tensor.dims();
    meta.backend = pten::TransToPtenBackend(framework_tensor.place());
    meta.type = pten::TransToPtenDataType(framework_tensor.type());
    meta.layout = pten::TransToPtenDataLayout(framework_tensor.layout());
    allocation = framework_tensor.Holder();
    auto impl_tensor =
        std::make_shared<TYPE>(std::move(tensor_meta), pten::TensorStatus());
    impl_tensor->ShareAllocation(allocation);
    this->set_impl(impl_tensor);
  }

 public:
  /** Part 11: Sync paddle::framework::Variable with pten::Tensor **/
  void SyncToTensor() {
    // Synchronize allocation only once.
    if (!this->initialized()) {
      // TODO(jiabin): Support selected rows later.
      PADDLE_ENFORCE(var_.IsInitialized(),
                     paddle::platform::errors::Fatal(
                         "Can not Sync EagerTensor %s whose "
                         "paddle::framework::Variable is not initialized!" %
                         name()));
      std::shared_ptr<paddle::memory::allocation::Allocation> allocation;
      if (var_.IsType<paddle::framework::LoDTensor>()) {
        SetImplWithLegacyTensor<paddle::framework::LoDTensor,
                                pten::DenseTensor>();
      } else if (var_.IsType<paddle::framework::Tensor>()) {
        SetImplWithLegacyTensor<paddle::framework::Tensor, pten::DenseTensor>();
      } else {
        PADDLE_THROW(
            paddle::platform::errors::Fatal("Unable to fetch underlying tensor "
                                            "from VarBase, only LoDTensor and "
                                            "Tensor are supported for now"));
      }
    }
  }

 private:
  std::shared_ptr<paddle::experimental::Tensor> tensor_ = nullptr;
  paddle::framework::Variable var_;
};
}  // namespace egr
