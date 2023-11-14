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
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";
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
  // TODO(liuzhenhai): add latter
  return SpmdInfo();
}
SpmdInfo ConcatInferSpmdDynamic(const std::vector<DistMetaTensor>& x,
                                const Scalar& axis) {
  return ConcatInferSpmd(x, axis.to<int32_t>());
}
}  // namespace distributed
}  // namespace phi
