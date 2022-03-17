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

/**
 * We now still need TensorWrapper and it is designed to Copy
 * tensor in autograd mode.
 *
 * Since in autograd usage, we need to pass autograd_meta to
 * backward computation however in tensor interface add to much
 * autograd_related method is not a good choice.
 *
 * In TensorWrapper we will keep autograd info to backward, only
 * for input var, but for output var it will only copy autograd
 * with no grad **/

#pragma once
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/utils.h"

namespace egr {
class TensorWrapper {
 public:
  TensorWrapper() = default;
  explicit TensorWrapper(const paddle::experimental::Tensor& tensor,
                         bool full_reserved = false,
                         bool no_need_buffer = false) {
    /**
     * Normally, we should fully reserved all non-output or non-leaf fwd tensor
     * here. And for fwd output tensor, we should not reserve its autogradmeta,
     * to avoid recursive depends on GradNodeBase
     * **/
    full_reserved_ = full_reserved;
    if (full_reserved_) {
      VLOG(6) << "Fully reserved tensor: " << tensor.name();
      intermidiate_tensor_ = tensor;

      // set inplace_version_snapshot_ according to tensor's current inplace
      // version.
      if (phi::DenseTensor::classof(tensor.impl().get())) {
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(tensor.impl().get());
        auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();
        inplace_version_snapshot_ = inplace_version_counter.CurrentVersion();

      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized tensor type for snapshoting inplace version."));
      }
      return;
    }

    // shallow copy tensor_impl here
    no_need_buffer_ = no_need_buffer;
    if (no_need_buffer) {
      if (phi::DenseTensor::classof(tensor.impl().get())) {
        // Only Copy Meta
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(tensor.impl().get());
        auto tw_dense_tensor = std::make_shared<phi::DenseTensor>();
        tw_dense_tensor->set_meta(dense_tensor->meta());
        intermidiate_tensor_.set_impl(tw_dense_tensor);
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized tensor type for no_need_buffer feature"));
      }
    } else {
      intermidiate_tensor_.set_impl(tensor.impl());
    }

    intermidiate_tensor_.set_name(tensor.name() + "@Saved");

    // If an output is marked "intermedaite", we won't create
    // autograd_meta for it.
    // In that case, simply skip OutRankInfo Copy
    if (EagerUtils::nullable_autograd_meta(tensor)) {
      out_rank_info_ = EagerUtils::OutRankInfo(tensor);
    }
  }

  paddle::experimental::Tensor recover(
      const std::shared_ptr<GradNodeBase>& grad_node) {
    VLOG(6) << "Recover tensor: " << intermidiate_tensor_.name()
            << " for wrapper";
    if (!intermidiate_tensor_.defined()) {
      VLOG(6) << "Return NULL tensor Here. ";
      return paddle::experimental::Tensor();
    }

    // if it's full_reserved just return the full copy of tensor
    if (full_reserved_) {
      check_inplace_version();
      return intermidiate_tensor_;
    } else {
      std::shared_ptr<GradNodeBase> new_grad_node = grad_node;
      auto p_ab_autograd_meta =
          std::make_shared<AutogradMeta>(Edge(new_grad_node, out_rank_info_));
      intermidiate_tensor_.set_autograd_meta(
          std::static_pointer_cast<paddle::experimental::AbstractAutogradMeta>(
              p_ab_autograd_meta));
      // check_inplace_version();
      return intermidiate_tensor_;
    }
  }

  void check_inplace_version() {
    if (no_need_buffer_) {
      VLOG(6) << "There's no need to check inplace_version because "
                 "no_need_buffer_ is true.";
      return;
    }
    if (phi::DenseTensor::classof(intermidiate_tensor_.impl().get())) {
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(intermidiate_tensor_.impl().get());
      auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();

      uint32_t current_inplace_version =
          inplace_version_counter.CurrentVersion();
      PADDLE_ENFORCE_EQ(
          current_inplace_version, inplace_version_snapshot_,
          paddle::platform::errors::PermissionDenied(
              "Tensor '%s' used in gradient computation has been "
              "modified by an inplace operation. "
              "Its version is %d but the expected version is %d. "
              "Please fix your code to void calling an inplace operator "
              "after using the Tensor which will used in gradient "
              "computation.",
              intermidiate_tensor_.name(), current_inplace_version,
              inplace_version_snapshot_));
      VLOG(6) << " The inplace_version_snapshot_ of Tensor '"
              << intermidiate_tensor_.name() << "' is [ "
              << inplace_version_snapshot_ << " ]";
    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Unrecognized tensor type for checking inplace version."));
    }
  }

  void clear() { intermidiate_tensor_.reset(); }

 private:
  bool full_reserved_ = false;
  bool no_need_buffer_ = false;
  std::pair<size_t, size_t> out_rank_info_;
  paddle::experimental::Tensor intermidiate_tensor_;
  uint32_t inplace_version_snapshot_ = 0;
};
}  // namespace egr
