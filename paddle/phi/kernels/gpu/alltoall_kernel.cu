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

#include "glog/logging.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace phi {

using phi::distributed::GetPartialTensor;

template <typename T, typename Context>
void AllToAllOpCUDAKernel(const Context& dev_ctx,
                          const DenseTensor& x_in,
                          int ring_id,
                          bool use_calc_stream,
                          DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
  auto x = &x_in;
  int send_numel = x->numel();
  ncclDataType_t dtype = phi::ToNCCLDataType(x->dtype());

  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      phi::errors::InvalidArgument(
          "The ring_id (%d) for alltoall op must be non-negative.", ring_id));
  auto place = dev_ctx.GetPlace();

  gpuStream_t stream = nullptr;
  paddle::platform::NCCLComm* comm = nullptr;
  phi::distributed::NCCLCommContext* comm_ctx = nullptr;
  int nranks = 0;

  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  if (FLAGS_dynamic_static_unified_comm) {
    PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                      true,
                      phi::errors::InvalidArgument(
                          "You choose to use new communication library by "
                          "setting environment "
                          "variable FLAGS_dynamic_static_unified_comm True. "
                          "But ring_id(%d) is "
                          "not found in comm_context_manager.",
                          std::to_string(ring_id)));
    comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
        comm_context_manager.Get(std::to_string(ring_id)));
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      phi::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
    stream = comm_ctx->GetStream();
    nranks = comm_ctx->GetSize();
    VLOG(3) << "new comm_context_manager has rid " << ring_id;
  } else {
    comm = paddle::platform::NCCLCommContext::Instance().Get(ring_id, place);
    stream = comm->stream();
    nranks = comm->nranks();
    VLOG(3) << "old NCCLCommContext has rid " << ring_id;
  }

  if (use_calc_stream) {
    // should ExecutionContext for calc stream.
    stream = dev_ctx.stream();
  }

  phi::DDim x_dims = x->dims();
  phi::DDim out_dims(x_dims);
  PADDLE_ENFORCE_EQ(
      x_dims[0] % nranks,
      0,
      phi::errors::InvalidArgument(
          "The first dimension size (%d) of the input tensor must be "
          "divisible by the number of ranks (%d).",
          x_dims[0],
          nranks));
  auto send_buf = x->data<T>();
  out->Resize(out_dims);
  auto recv_buf = dev_ctx.template Alloc<T>(out);
  size_t offset = 0;
  send_numel /= nranks;
  if (comm_ctx) {
    comm_ctx->GroupStart();
    for (auto i = 0; i < nranks; ++i) {
      auto send_buf = GetPartialTensor(*x, offset, send_numel);
      comm_ctx->Send(send_buf, send_numel, i, stream);
      auto recv_buf = GetPartialTensor(*out, offset, send_numel);
      comm_ctx->Recv(&recv_buf, send_numel, i, stream);
      offset += send_numel;
    }
    comm_ctx->GroupEnd();
    VLOG(3) << "new comm_context_manager has rid " << ring_id;
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGroupStart());
    for (auto i = 0; i < nranks; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclSend(
          send_buf + offset, send_numel, dtype, i, comm->comm(), stream));
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRecv(
          recv_buf + offset, send_numel, dtype, i, comm->comm(), stream));
      offset += send_numel;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGroupEnd());
    VLOG(3) << "old NCCLCommContext has rid " << ring_id;
  }
#else
  PADDLE_THROW(phi::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
  PADDLE_THROW(
      phi::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(alltoall,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllToAllOpCUDAKernel,
                   float,
                   double,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                   phi::dtype::bfloat16,
#endif
                   int,
                   int64_t,
                   phi::dtype::float16) {
}
