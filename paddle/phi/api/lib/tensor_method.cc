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

#include "paddle/phi/api/include/tensor.h"

#include "paddle/phi/api/lib/ext_compat_utils.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_base.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace experimental {
// declare cast api
Tensor cast(const Tensor &x, DataType out_dtype);
Tensor copy_to(const Tensor &x, Backend backend, bool blocking);

Tensor Tensor::cast(DataType target_type) const {
  return experimental::cast(*this, target_type);
}

Tensor Tensor::copy_to(Backend backend, bool blocking) const {
  return experimental::copy_to(*this, backend, blocking);
}

template <typename T>
Tensor Tensor::copy_to(const PlaceType &target_place) const {
  LOG(WARNING) << "The Tensor's `copy_to` method is deprecated since version "
                  "2.3, and will be removed in version 2.4, please use "
                  "`copy_to` method without template argument instead. "
                  "reason: copying a Tensor to another device does not need "
                  "to specify the data type template argument.";
  return copy_to(ConvertExtPlaceToBackend(target_place), /*blocking=*/false);
}

template PADDLE_API Tensor
Tensor::copy_to<float>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<double>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int64_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int32_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<uint8_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int8_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int16_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<bool>(const PlaceType &target_place) const;
template PADDLE_API Tensor Tensor::copy_to<phi::dtype::complex<float>>(
    const PlaceType &target_place) const;
template PADDLE_API Tensor Tensor::copy_to<phi::dtype::complex<double>>(
    const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<phi::dtype::float16>(const PlaceType &target_place) const;

void Tensor::copy_(const Tensor &src,
                   const phi::Place &target_place,
                   bool blocking) {
  if (!src.is_initialized()) {
    VLOG(8) << "Src is empty, skip copy";
    return;
  }
  // Prepare copy kernel key and outputs
  auto kernel_key_set = ParseKernelKeyByInputArgs(src);
  KernelType kernel_type = ParseKernelTypeByInputArgs(src);
  VLOG(3) << "Deep copy Tensor from " << src.name() << " to " << name();
  if (is_initialized()) {
    PADDLE_ENFORCE_EQ(dtype(),
                      src.dtype(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s, "
                          "Tensor Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(target_place,
                      inner_place(),
                      platform::errors::PreconditionNotMet(
                          "Place is different of dst tensor and args %s, which "
                          "current tensor holds %s "
                          "Copy cannot be performed!",
                          target_place.DebugString(),
                          inner_place().DebugString()));
    kernel_key_set.backend_set =
        kernel_key_set.backend_set |
        BackendSet(phi::TransToPhiBackend(inner_place()));
  } else {
    // Deep Copy AutoGrad info from src to self.
    *autograd_meta_ = *(src.autograd_meta_);
  }

  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  auto *dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (kernel_type == KernelType::DENSE_TENSOR_KENREL) {
    auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "copy", {kernel_backend, kernel_layout, kernel_data_type});
    VLOG(6) << "copy API kernel key: " << kernel_key;
    VLOG(6) << "copy API kernel: " << kernel;
    using kernel_signature = void (*)(const platform::DeviceContext &,
                                      const phi::DenseTensor &,
                                      phi::Place,
                                      bool,
                                      phi::DenseTensor *);
    SetKernelOutput(kernel_backend, this);
    phi::MetaTensor meta_out(impl_.get());
    phi::UnchangedInferMeta(
        MakeMetaTensor(
            *(std::static_pointer_cast<phi::DenseTensor>(src.impl_))),
        &meta_out);
    auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    (*kernel_fn)(*dev_ctx,
                 (*(std::static_pointer_cast<phi::DenseTensor>(src.impl_))),
                 target_place,
                 blocking,
                 static_cast<phi::DenseTensor *>(impl_.get()));
  } else if (kernel_type == KernelType::SELECTED_ROWS_KENREL) {
    auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "copy_sr", {kernel_backend, kernel_layout, kernel_data_type});
    VLOG(6) << "copy API kernel key: " << kernel_key;
    VLOG(6) << "copy API kernel: " << kernel;
    using kernel_signature = void (*)(const platform::DeviceContext &,
                                      const phi::SelectedRows &,
                                      phi::Place,
                                      bool,
                                      phi::SelectedRows *);
    SetSelectedRowsKernelOutput(kernel_backend, this);
    phi::MetaTensor meta_out(impl_.get());
    phi::UnchangedInferMeta(
        MakeMetaTensor(
            *(std::static_pointer_cast<phi::SelectedRows>(src.impl_))),
        &meta_out);
    auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    (*kernel_fn)(*dev_ctx,
                 (*(std::static_pointer_cast<phi::SelectedRows>(src.impl_))),
                 target_place,
                 blocking,
                 static_cast<phi::SelectedRows *>(impl_.get()));
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "We currently only support dense tensor copy for now and if u need to "
        "copy selected rows please raise a issue."));
  }
}

}  // namespace experimental
}  // namespace paddle
