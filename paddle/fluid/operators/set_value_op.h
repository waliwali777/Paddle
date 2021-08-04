//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/slice_utils.h"
#include "paddle/fluid/operators/strided_slice_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Variable = framework::Variable;
using LoDTensorArray = framework::LoDTensorArray;
using DDim = framework::DDim;

void get_offsets(const DDim& big_dim, const DDim& small_dim, DDim start_offset,
                 int cur_dim, std::vector<DDim>* offsets) {
  if (cur_dim == big_dim.size()) {
    offsets->push_back(start_offset);
    return;
  }
  if (small_dim[cur_dim] == big_dim[cur_dim]) {
    get_offsets(big_dim, small_dim, start_offset, cur_dim + 1, offsets);
  } else {
    for (int i = 0; i < big_dim[cur_dim]; i++) {
      get_offsets(big_dim, small_dim, start_offset, cur_dim + 1, offsets);
      start_offset[cur_dim] += 1;
    }
  }
}

inline std::string GetValueName(framework::proto::VarType::Type data_type) {
  std::string value_name;
  switch (data_type) {
    case framework::proto::VarType::INT32:
      value_name = "int32_values";
      break;
    case framework::proto::VarType::INT64:
      value_name = "int64_values";
      break;
    case framework::proto::VarType::FP32:
      value_name = "fp32_values";
      break;
    case framework::proto::VarType::FP64:
      value_name = "fp64_values";
      break;
    case framework::proto::VarType::BOOL:
      value_name = "bool_values";
      break;

    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type(code %d) for SetValue operator, only "
          "supports bool, int32, float32 and int64.",
          data_type));
  }
  return value_name;
}

template <typename DeviceContext, typename T>
class SetValueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const int rank = ctx.Input<framework::LoDTensor>("Input")->dims().size();

    // TODO(liym27): A more elegent code to do this. C++ has to make template
    //  integer as constant, but we had better have alternative writing in the
    //  future.
    switch (rank) {
      case 1:
        SetValueCompute<1>(ctx);
        break;
      case 2:
        SetValueCompute<2>(ctx);
        break;
      case 3:
        SetValueCompute<3>(ctx);
        break;
      case 4:
        SetValueCompute<4>(ctx);
        break;
      case 5:
        SetValueCompute<5>(ctx);
        break;
      case 6:
        SetValueCompute<6>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input should be less than 7, but received %d.", rank));
    }
  }

 private:
  template <size_t D>
  void SetValueCompute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::LoDTensor>("Input");
    auto* value_tensor = ctx.Input<framework::LoDTensor>("ValueTensor");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto starts_tensor_list =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto ends_tensor_list = ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto steps_tensor_list =
        ctx.MultiInput<framework::Tensor>("StepsTensorList");

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto steps = ctx.Attr<std::vector<int64_t>>("steps");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto decrease_axes = ctx.Attr<std::vector<int64_t>>("decrease_axes");

    auto dtype = in->type();
    if (!starts_tensor_list.empty()) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }
    if (!ends_tensor_list.empty()) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }
    if (!steps_tensor_list.empty()) {
      steps = GetDataFromTensorList<int64_t>(steps_tensor_list);
    }

    auto in_dims = in->dims();
    CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends, &steps);
    auto slice_dims = GetSliceDims(in_dims, axes, starts, ends, &steps);
    auto decrease_slice_dims = GetDecreasedDims(slice_dims, decrease_axes);

    auto place = ctx.GetPlace();
    auto& eigen_place =
        *ctx.template device_context<DeviceContext>().eigen_device();

    // Here copy data from input to avoid data loss at PE and Graph level.
    // TODO(liym27): Speed up in the future version.
    // - Q: Why don't call ShareDataWith to speed up?
    // - A: Because it's not supported to ShareDataWith on OP's input and output
    // https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-and-ShareBufferWith-are-prohibited-in-OP
    // - Q: Why don't delete Input, after all, the input and output are the same
    // Tensor at program level?
    // - A: If deleting Input, the graph will be complex, such as there will
    // be two ops points to the output in graph: op1 -> output <- set_value.
    // In this case, we have to find a way to handle the running order of
    // set_value is what we want.
    TensorCopy(*in, place, out);

    Tensor slice_tensor(dtype), pad_tensor(dtype);
    slice_tensor.mutable_data<T>(slice_dims, place);
    pad_tensor.mutable_data<T>(in_dims, place);

    auto pad_e = framework::EigenTensor<T, D>::From(pad_tensor, in_dims);
    auto out_e = framework::EigenTensor<T, D>::From(*out);
    auto slice_e = framework::EigenTensor<T, D>::From(slice_tensor, slice_dims);

    // Step 1: Set the value of out at `_index` to zero
    slice_e.device(eigen_place) = slice_e.constant(T(0));

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    for (size_t i = 0; i < D; ++i) {
      starts_indices[i] = 0;
      ends_indices[i] = slice_dims[i];
      strides_indices[i] = 1;
    }
    for (size_t i = 0; i < axes.size(); i++) {
      int axis_index = axes[i];
      starts_indices[axis_index] = starts[i];
      ends_indices[axis_index] = ends[i];
      strides_indices[axis_index] = steps[i];
    }

    out_e.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(eigen_place) = slice_e;

    // Step 2: Set a tensor with the same shape as out tensor. And its data at
    // '_index' is the same as value_tensor, and data out of '_index' to zero

    // - Step 2.1 Set slice tensor with value

    // NOTE(liym27): [ Why resize slice_tensor here? ]
    // A: When do broadcasting on slice_tensor and value_tensor, the shape of
    // slice_tensor should be decreased dims.
    // e.g.
    //  x[:,0] = value_tensor
    // x's shape = [3, 4], value_tensor's shape = [3]
    // We get slice_dims = [3, 1],  decrease_slice_dims = [3]
    // If do broadcasting on Tensor with shape [3, 1] and [3], the result's
    // shape is [3, 3], which cross the border;
    // If do broadcasting on Tensor with shape [3] and [3], the result's shape
    // is [3], which is right.

    slice_tensor.Resize(decrease_slice_dims);
    if (value_tensor != nullptr) {
      // ElementwiseComputeEx can do broadcasting
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_tensor, value_tensor, -1, SubFunctor<T>(), &slice_tensor);
    } else {
      Tensor value_t(dtype);
      auto value_dims = framework::make_ddim(shape);
      value_t.mutable_data<T>(value_dims, place);
      auto value_name = GetValueName(dtype);
      CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
      value_t.Resize(value_dims);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_tensor, &value_t, -1, SubFunctor<T>(), &slice_tensor);
    }
    slice_tensor.Resize(slice_dims);

    // - Step 2.2 Pad slice tensor with 0
    pad_e.device(eigen_place) = pad_e.constant(T(0));
    pad_e.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(eigen_place) = slice_e;

    // Step 3: Set out tensor with value_tensor
    out_e.device(eigen_place) = out_e - pad_e;
  }
};

template <typename DeviceContext, typename T>
class SetValueGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims().size();

    switch (rank) {
      case 1:
        SetValueGradCompute<1>(ctx);
        break;
      case 2:
        SetValueGradCompute<2>(ctx);
        break;
      case 3:
        SetValueGradCompute<3>(ctx);
        break;
      case 4:
        SetValueGradCompute<4>(ctx);
        break;
      case 5:
        SetValueGradCompute<5>(ctx);
        break;
      case 6:
        SetValueGradCompute<6>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of set_value_grad's input should be less than 7, but "
            "received %d.",
            rank));
    }
  }

 private:
  template <size_t D>
  void SetValueGradCompute(const framework::ExecutionContext& context) const {
    auto starts_int = context.Attr<std::vector<int>>("starts");
    auto ends_int = context.Attr<std::vector<int>>("ends");
    auto steps_int = context.Attr<std::vector<int>>("steps");

    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
    std::vector<int64_t> steps(steps_int.begin(), steps_int.end());

    auto axes = context.Attr<std::vector<int>>("axes");

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto steps_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto reverse_axis = Eigen::array<bool, D>();

    auto list_new_ends_tensor =
        context.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        context.MultiInput<framework::Tensor>("StartsTensorList");
    auto list_new_steps_tensor =
        context.MultiInput<framework::Tensor>("StepsTensorList");

    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (context.HasInput("StartsTensor")) {
      auto* starts_tensor = context.Input<framework::Tensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
    } else if (context.HasInput("EndsTensor")) {
      auto* ends_tensor = context.Input<framework::Tensor>("EndsTensor");
      ends = GetDataFromTensor<int64_t>(ends_tensor);
    }

    if (list_new_steps_tensor.size() > 0) {
      steps = GetDataFromTensorList<int64_t>(list_new_steps_tensor);
    } else if (context.HasInput("StepsTensor")) {
      auto* steps_tensor = context.Input<framework::Tensor>("StepsTensor");
      steps = GetDataFromTensor<int64_t>(steps_tensor);
    }

    auto in = context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto grad_value = context.Output<framework::Tensor>(
        framework::GradVarName("ValueTensor"));
    auto grad_input =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    auto in_dims = in->dims();

    std::vector<int> decrease_axis;
    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
    StridedSliceOutDims(starts, ends, steps, axes, infer_flags, in_dims,
                        decrease_axis, out_dims_vector.data(), axes.size(),
                        false);

    framework::DDim out_dims(framework::make_ddim(out_dims_vector));

    std::vector<int> reverse_vector(starts.size(), 0);
    StridedSliceFunctor(starts.data(), ends.data(), steps.data(), axes.data(),
                        reverse_vector.data(), in_dims, infer_flags,
                        decrease_axis, starts.size());

    for (size_t axis = 0; axis < D; axis++) {
      starts_indices[axis] = 0;
      ends_indices[axis] = out_dims[axis];
      steps_indices[axis] = 1;
      reverse_axis[axis] = false;
    }

    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices[axis_index] = starts[axis];
      ends_indices[axis_index] = ends[axis];
      steps_indices[axis_index] = steps[axis];
      reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
    }

    bool need_reverse = false;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        need_reverse = true;
        break;
      }
    }

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    // Set gradient of `Input`
    TensorCopy(*in, context.GetPlace(), grad_input);

    if (!grad_value->IsInitialized()) {
      grad_value->mutable_data<T>(context.GetPlace());
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();

    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, grad_value, static_cast<T>(0));

    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto grad_value_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *grad_value, out_dims);

    auto grad_input_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *grad_input);

    grad_input_t.stridedSlice(starts_indices, ends_indices, steps_indices)
        .setZero();

    if (grad_value->dims() == out_dims) {
      if (need_reverse) {
        framework::Tensor tmp;
        tmp.mutable_data<T>(out_dims, context.GetPlace());
        auto tmp_t = framework::EigenTensor<T, D, Eigen::RowMajor,
                                            Eigen::DenseIndex>::From(tmp);
        tmp_t.device(place) =
            in_t.stridedSlice(starts_indices, ends_indices, steps_indices);
        grad_value_t.device(place) = tmp_t.reverse(reverse_axis);
      } else {
        grad_value_t.device(place) =
            in_t.stridedSlice(starts_indices, ends_indices, steps_indices);
      }
    } else {
      int out_dims_size = out_dims.size();
      auto grad_value_dims = grad_value->dims();
      auto fake_grad_value_dims = out_dims;

      // Create an extented shape according to the rules of broadcast.
      auto grad_value_dims_size = grad_value_dims.size();
      auto dim_one = out_dims_size;
      for (int i = out_dims_size - 1; i >= 0; i--) {
        if (out_dims[i] == 1) {
          dim_one = i;
        } else {
          break;
        }
      }
      for (int i = 0; i < out_dims_size; i++) {
        if (i < dim_one - grad_value_dims_size || dim_one <= i) {
          fake_grad_value_dims[i] = 1;
        } else {
          auto index_grad = i - (dim_one - grad_value_dims_size);
          fake_grad_value_dims[i] = grad_value_dims[index_grad];
          PADDLE_ENFORCE_EQ((out_dims[i] == grad_value_dims[index_grad]) ||
                                (grad_value_dims[index_grad] == 1),
                            true, platform::errors::InvalidArgument(
                                      "An error occurred while calculating %s: "
                                      "[%s] can not be accumulated into [%s].",
                                      framework::GradVarName("ValueTensor"),
                                      out_dims, grad_value_dims));
        }
      }

      VLOG(3) << "Dimensions of " << framework::GradVarName("ValueTensor")
              << "([" << grad_value_dims << "])is broadcasted into ["
              << fake_grad_value_dims << "].";

      auto extent = Eigen::DSizes<Eigen::DenseIndex, D>();
      auto offset = out_dims;
      for (int i = 0; i < out_dims_size; i++) {
        offset[i] = 0;
        extent[i] = fake_grad_value_dims[i];
      }
      std::vector<DDim> offsets;
      get_offsets(out_dims, fake_grad_value_dims, offset, 0, &offsets);
      framework::Tensor tmp;
      tmp.mutable_data<T>(out_dims, context.GetPlace());
      auto tmp_t = framework::EigenTensor<T, D, Eigen::RowMajor,
                                          Eigen::DenseIndex>::From(tmp);

      tmp_t.device(place) =
          in_t.stridedSlice(starts_indices, ends_indices, steps_indices);

      if (need_reverse) {
        tmp_t.device(place) = tmp_t.reverse(reverse_axis);
      }
      for (auto offset : offsets) {
        grad_value_t.device(place) =
            grad_value_t +
            tmp_t.slice(framework::EigenDim<D>::From(offset), extent);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
