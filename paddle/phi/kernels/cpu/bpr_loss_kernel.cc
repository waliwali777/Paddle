// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include "paddle/common/enforce.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
    const T kApproInf = 1e20;
    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

template <typename T, typename Context>
void BprLossOpKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& label,
                     DenseTensor* out) {
  ctx.template Alloc<T>(out);
  int rank = input.dims().size();

  phi::DenseTensor x_2d = phi::ReshapeToMatrix(input, rank - 1);
  phi::DenseTensor labels_2d = phi::ReshapeToMatrix(label, rank - 1);
  phi::DenseTensor y_2d = phi::ReshapeToMatrix(*out, rank - 1);

  const phi::DenseTensor* logits = &x_2d;
  const phi::DenseTensor* labels = &labels_2d;
  phi::DenseTensor* out_y_2d = &y_2d;

  const int step_size = logits->dims()[0];
  const int class_num = logits->dims()[1];
  const T* logits_data = logits->data<T>();
  T* loss_data = out_y_2d->data<T>();

  const int64_t* label_data = labels->data<int64_t>();
  for (int i = 0; i < step_size; ++i) {
    int lbl_pos = label_data[i];
    PADDLE_ENFORCE_GE(
        lbl_pos,
        0,
        phi::errors::InvalidArgument("label data %d is illegal.", lbl_pos));
    PADDLE_ENFORCE_LT(
        lbl_pos,
        class_num,
        phi::errors::InvalidArgument("label data %d is illegal.", lbl_pos));
    int index_pos = i * class_num + lbl_pos;
    T sum = static_cast<T>(0);
    for (int j = 0; j < class_num; j++) {
      if (j == lbl_pos) continue;
      int index_neg = i * class_num + j;
      sum += TolerableValue<T>()(-std::log(
          1.0f + TolerableValue<T>()(std::exp(logits_data[index_neg] -
                                              logits_data[index_pos]))));
    }
    loss_data[i] = -sum / (class_num - 1);
  }
}

template <typename T, typename Context>
void BprLossGradientOpKernel(const Context& ctx,
                             const DenseTensor& input,
                             const DenseTensor& label,
                             const DenseTensor& out_grad,
                             DenseTensor* x_grad) {
  T* dx_data = ctx.template Alloc<T>(x_grad);

  const size_t step_size = static_cast<size_t>(input.dims()[0]);
  const size_t num_classes = static_cast<size_t>(input.dims()[1]);
  const T* dy_data = out_grad.data<T>();
  const T* x_data = input.data<T>();
  const int64_t* label_data = label.data<int64_t>();

  for (size_t sample_id = 0; sample_id < step_size; sample_id++) {
    for (size_t x_offset = sample_id * num_classes;
         x_offset < (sample_id + 1) * num_classes;
         x_offset++) {
      dx_data[x_offset] = static_cast<T>(0);
    }
    auto p_index = sample_id * num_classes + label_data[sample_id];
    for (size_t ni = 0; ni < num_classes; ni++) {
      if (label_data[sample_id] == static_cast<int>(ni)) continue;
      auto n_index = sample_id * num_classes + ni;
      auto grad_ = -dy_data[sample_id] /
                   ((num_classes - 1) *
                    (1.0f + TolerableValue<T>()(
                                std::exp(x_data[p_index] - x_data[n_index]))));
      dx_data[p_index] += grad_;
      dx_data[n_index] -= grad_;
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    bpr_loss, CPU, ALL_LAYOUT, phi::BprLossOpKernel, float, double) {}

PD_REGISTER_KERNEL(bpr_loss_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::BprLossGradientOpKernel,
                   float,
                   double) {}
