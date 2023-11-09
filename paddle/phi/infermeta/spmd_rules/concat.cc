/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/concat.h"

#include <limits>
#include <set>

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

std::tuple<std::string, std::string> FillConcatNotation(int64_t n_axis,
                                                        int64_t concat_axis) {
  PADDLE_ENFORCE_EQ(
      n_axis > concat_axis, true, phi::errors::InvalidArgument(""));
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";
  PADDLE_ENFORCE_EQ(alphabet.size() > static_cast<size_t>(n_axis),
                    true,
                    phi::errors::InvalidArgument("n_axis %d", n_axis));
  std::string all_axis = alphabet.substr(0, n_axis);
  std::string align_axis =
      std::string(all_axis.begin(), all_axis.begin() + concat_axis) +
      std::string(all_axis.begin() + concat_axis + 1, all_axis.end());
  return {all_axis, align_axis};
}

SpmdInfo ConcatInferSpmd(const std::vector<DistMetaTensor>& x, int axis) {
  /*
# paddle.concat requires all tensors must either have the same shape (except
# in the concatenating dimension) or be "empty". "Empty" here strictly means
# tensor.shape is torch.Size([0]). When tensor.ndim > 1, it will be treated
# as a non-empty tensor and the shape must match on non-cat dimensions.
 */

  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return phi::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  int64_t ndim = static_cast<int64_t>(tensor_shapes[non_empty_index].size());
  // normlize dim
  auto dim = axis < 0 ? ndim + axis : axis;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });

  std::string all_aixs;
  std::string align_axis;
  std::tie(all_aixs, align_axis) = FillConcatNotation(ndim, dim);
  std::vector<std::string> axis_names(input_attrs.size(), all_aixs);
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, align_axis, true);

  return {{input_attrs}, {input_attrs[non_empty_index]}};
}

SpmdInfo ConcatInferSpmdReverse(const std::vector<DistMetaTensor>& x,
                                const DistMetaTensor& output,
                                int axis) {
  auto out_dist_attr = output.dist_attr();
  out_dist_attr = UnShardTensorDim(out_dist_attr, axis);
  auto n_inputs = x.size();
  TensorDistAttr input_attr = CopyTensorDistAttrForOutput(out_dist_attr, false);
  const auto& input_dim_mapping = out_dist_attr.dims_mapping();
  input_attr.set_dims_mapping(input_dim_mapping);
  std::vector<TensorDistAttr> input_attrs(n_inputs, input_attr);
  return {{input_attrs}, {output.dist_attr()}};
}

SpmdInfo ConcatInferSpmdDynamic(const std::vector<DistMetaTensor>& x,
                                const Scalar& axis) {
  return ConcatInferSpmd(x, axis.to<int32_t>());
}

std::string FillStackNotation(int64_t n_axis) {
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";
  PADDLE_ENFORCE_EQ(alphabet.size() > static_cast<size_t>(n_axis),
                    true,
                    phi::errors::InvalidArgument("n_axis %d", n_axis));
  std::string all_axis = alphabet.substr(0, n_axis);
}

SpmdInfo StackInferSpmd(const std::vector<DistMetaTensor>& x, int axis) {
  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return phi::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });

  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  auto ndim = tensor_shapes[non_empty_index].size();
  // normlize dim
  auto dim = axis < 0 ? static_cast<int64_t>(ndim) + axis : axis;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });
  std::string notation = FillStackNotation(ndim);
  std::vector<std::string> axis_names(input_attrs.size(), notation);
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, notation, true);

  TensorDistAttr output_attr =
      CopyTensorDistAttrForOutput(input_attrs[non_empty_index], false);
  std::vector<int64_t> dim_mapping(ndim + 1, -1);
  const auto& input_dim_mapping = input_attrs[non_empty_index].dims_mapping();
  for (size_t i = 0; i < ndim; i++) {
    size_t out_index = i < static_cast<size_t>(dim) ? i : (i + 1);
    dim_mapping[out_index] = input_dim_mapping[i];
  }
  output_attr.set_dims_mapping(dim_mapping);
  return {{input_attrs}, {output_attr}};
}

SpmdInfo StackInferSpmdReverse(const std::vector<DistMetaTensor>& x,
                               const DistMetaTensor& output,
                               int axis) {
  auto out_dist_attr = output.dist_attr();
  out_dist_attr = UnShardTensorDim(out_dist_attr, axis);
  auto n_inputs = x.size();
  TensorDistAttr input_attr = CopyTensorDistAttrForOutput(out_dist_attr, false);
  auto ndim = output.dims().size();
  std::vector<int64_t> dim_mapping(ndim - 1, -1);
  const auto& input_dim_mapping = out_dist_attr.dims_mapping();
  for (size_t i = 0; i < static_cast<size_t>(ndim - 1); i++) {
    size_t out_index = i < static_cast<size_t>(axis) ? i : (i + 1);
    dim_mapping[i] = input_dim_mapping[out_index];
  }
  input_attr.set_dims_mapping(dim_mapping);
  std::vector<TensorDistAttr> input_attrs(n_inputs, input_attr);
  return {{input_attrs}, {output.dist_attr()}};
}

SpmdInfo ConcatGradInferSpmdDynamic(const std::vector<DistMetaTensor>& x,
                                    const DistMetaTensor& output_grad,
                                    const Scalar& axis) {
  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return phi::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  int64_t ndim = static_cast<int64_t>(tensor_shapes[non_empty_index].size());
  auto dim = axis.to<int64_t>();
  // normlize dim
  dim = dim < 0 ? ndim + dim : dim;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });
  input_attrs.push_back(output_grad.dist_attr());
  std::string all_aixs;
  std::string align_axis;
  std::tie(all_aixs, align_axis) = FillConcatNotation(ndim, dim);
  std::vector<std::string> axis_names(input_attrs.size(), all_aixs);
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, align_axis, true);
  auto output_grad_attr = input_attrs.back();
  input_attrs.pop_back();
  std::vector<TensorDistAttr> inputs_grad = input_attrs;
  return {{input_attrs, output_grad_attr}, {inputs_grad}};
}

std::tuple<std::string, std::string> FillStackGradNotation(int64_t n_axis,
                                                           int64_t stack_dim) {
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";
  PADDLE_ENFORCE_EQ(alphabet.size() > static_cast<size_t>(n_axis + 1),
                    true,
                    phi::errors::InvalidArgument("n_axis %d", n_axis));
  std::string input_axis = alphabet.substr(0, n_axis);
  std::string output_axis = input_axis.substr(0, stack_dim) +
                            alphabet[stack_dim] + input_axis.substr(stack_dim);
}

SpmdInfo StackGradInferSpmd(const std::vector<DistMetaTensor>& x,
                            const DistMetaTensor& output_grad,
                            int axis) {
  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return phi::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  int64_t ndim = static_cast<int64_t>(tensor_shapes[non_empty_index].size());
  int64_t dim = axis;
  // normlize dim
  dim = dim < 0 ? ndim + dim : dim;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });
  input_attrs.push_back(output_grad.dist_attr());
  std::string inputs_axis;
  std::string output_axis;
  std::tie(inputs_axis, output_axis) = FillStackGradNotation(ndim, dim);
  std::vector<std::string> axis_names(input_attrs.size(), inputs_axis);
  axis_names.push_back((output_axis));
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, inputs_axis, true);
  auto output_grad_attr = input_attrs.back();
  input_attrs.pop_back();
  std::vector<TensorDistAttr> inputs_grad = input_attrs;
  return {{input_attrs, output_grad_attr}, {inputs_grad}};
}

}  // namespace distributed
}  // namespace phi
