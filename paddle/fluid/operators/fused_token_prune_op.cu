/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused_token_prune_op.cu.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
class FusedTokenPruneOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* attn = context.Input<Tensor>("Attn");
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* mask = context.Input<Tensor>("Mask");
    Tensor* out_slimmed_x = context.Output<Tensor>("SlimmedX");
    auto factor = context.template Attr<float>("factor");
    auto* out_slimmed_x_data =
        out_slimmed_x->mutable_data<T>(context.GetPlace());

    Tensor attn_tmp;
    auto attn_dims = attn->dims();
    attn_tmp.Resize(attn_dims);
    auto* attn_tmp_data = attn_tmp.mutable_data<T>(context.GetPlace());

    std::vector<const Tensor*> ins;
    std::vector<Tensor*> outs;
    ins.emplace_back(attn);
    ins.emplace_back(mask);
    outs.emplace_back(&attn_tmp);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        context.cuda_device_context(), ins, &outs, -1, AttnMaskFunctor<T>());

    // VLOG(4) << "attn after mask = " << attn_tmp;
    Tensor attn_by;
    attn_by.Resize({attn_dims[0], attn_dims[3]});
    auto* attn_by_data = attn_by.mutable_data<T>(context.GetPlace());
    const std::vector<int64_t> reduce_dims{1, 2};
    phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
        context.cuda_device_context(), attn_tmp, false, reduce_dims, false,
        attn_by.dtype(), &attn_by);

    // VLOG(4) << "attn after sum reduce = " << attn_by;

    Tensor attn_by_indices;
    attn_by_indices.Resize(attn_by.dims());
    auto* attn_by_indices_data =
        attn_by_indices.mutable_data<int>(context.GetPlace());

    int grid_size = attn_dims[0], block_size = ComputeBlockSize(attn_dims[3]);
    FillIndex<<<grid_size, block_size, 0,
                context.cuda_device_context().stream()>>>(
        attn_by_indices_data, attn_dims[0], attn_dims[3]);

    // VLOG(4) << "before argsort attn indices = " << attn_by_indices;
    SlicedArgsort<
        T><<<grid_size, 1, 0, context.cuda_device_context().stream()>>>(
        attn_by_data, attn_by_indices_data, attn_dims[0], attn_dims[3]);
    // VLOG(4) << "after argsort attn indices = " << attn_by_indices;

    int slimmed_x_len = attn_dims[3] * factor;
    Tensor slimmed_indices =
        phi::funcs::Slice<int>(context.cuda_device_context(), attn_by_indices,
                               {1}, {0}, {slimmed_x_len});
    // VLOG(4) << "after slice attn indices = " << slimmed_indices;

    auto x_dims = x->dims();
    block_size = ComputeBlockSize(slimmed_x_len);
    TakeAlongAxis<T><<<grid_size, block_size, 0,
                       context.cuda_device_context().stream()>>>(
        x->data<T>(), out_slimmed_x_data, slimmed_indices.data<int>(),
        attn_dims[0], attn_dims[3], slimmed_x_len, x_dims[2]);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_token_prune,
                        ops::FusedTokenPruneOpCUDAKernel<float>,
                        ops::FusedTokenPruneOpCUDAKernel<double>);
