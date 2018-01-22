/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/data_type_transform.h"

#include "paddle/framework/selected_rows.h"
#include "paddle/platform/transform.h"

namespace paddle {
namespace framework {

template <typename InType, typename OutType>
struct CastDataTypeFunctor {
  HOSTDEVICE inline OutType operator()(InType in) const {
    return static_cast<OutType>(in);
  }
};

template <typename InType>
struct CastDataType {
  CastDataType(const framework::Tensor& in, framework::Tensor* out,
               const platform::DeviceContext* ctx)
      : in_(in), out_(out), ctx_(ctx) {}
  const framework::Tensor in_;
  framework::Tensor* out_;
  const platform::DeviceContext* ctx_;

  template <typename OutType>
  void operator()() {
    auto* in_begin = in_.data<InType>();
    auto* in_end = in_begin + in_.numel();
    auto* out_begin = out_->mutable_data<OutType>(in_.place());

    if (platform::is_cpu_place(in_.place())) {
      platform::Transform<platform::CPUDeviceContext> trans;
      auto* context = static_cast<const platform::CPUDeviceContext*>(ctx_);
      trans(*context, in_begin, in_end, out_begin,
            CastDataTypeFunctor<InType, OutType>());
    } else {
      // TODO(dzhwinter): enhance Copy CPU<->GPU with different data type?
      PADDLE_THROW("Unsupport CPU <-> GPU!");
    }
  }
};

void TransDataType(const OpKernelType& kernel_type_for_var,
                   const OpKernelType& expected_kernel_type, const Tensor& in,
                   Tensor* out) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  out->Resize(in.dims());
  auto src_type = kernel_type_for_var.data_type_;
  auto dst_type = expected_kernel_type.data_type_;
  auto ctx = pool.Get(in.place());

  switch (src_type) {
    case proto::DataType::FP32:
      framework::VisitDataType(dst_type, CastDataType<float>(in, out, ctx));
      break;
    case proto::DataType::FP64:
      framework::VisitDataType(dst_type, CastDataType<double>(in, out, ctx));
      break;
    case proto::DataType::INT32:
      framework::VisitDataType(dst_type, CastDataType<int>(in, out, ctx));
      break;
    case proto::DataType::INT64:
      framework::VisitDataType(dst_type, CastDataType<int64_t>(in, out, ctx));
      break;
    case proto::DataType::BOOL:
      framework::VisitDataType(dst_type, CastDataType<bool>(in, out, ctx));
      break;
    default:
      PADDLE_THROW("Not support type %d", src_type);
  }
}

}  // namespace framework
}  // namespace paddle
