/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {

enum ArgMinMaxType { kArgMin, kArgMax };

template <typename DeviceContext, typename T, typename Tout, int64_t Rank,
          ArgMinMaxType argMinMaxValue>
struct ArgMinMaxFunctor {};

#define DECLARE_ARG_MIN_MAX_FUNCTOR(eigen_op_type, enum_argminmax_value)     \
  template <typename DeviceContext, typename T, typename Tout, int64_t Rank> \
  struct ArgMinMaxFunctor<DeviceContext, T, Tout, Rank,                      \
                          enum_argminmax_value> {                            \
    void operator()(const DeviceContext& ctx, const framework::Tensor& in,   \
                    framework::Tensor& out, int64_t axis) {                  \
      auto in_eigen = framework::EigenTensor<T, Rank>::From(in);             \
      auto out_eigen = framework::EigenTensor<Tout, Rank - 1>::From(out);    \
      out_eigen.device(*(ctx.eigen_device())) =                              \
          in_eigen.eigen_op_type(axis).template cast<Tout>();                \
    }                                                                        \
  }

DECLARE_ARG_MIN_MAX_FUNCTOR(argmin, ArgMinMaxType::kArgMin);
DECLARE_ARG_MIN_MAX_FUNCTOR(argmax, ArgMinMaxType::kArgMax);

template <typename DeviceContext, typename T, typename Tout,
          ArgMinMaxType EnumArgMinMaxValue>
class ArgMinMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& x = *(ctx.Input<framework::Tensor>("X"));
    auto& out = *(ctx.Output<framework::Tensor>("Out"));
    out.mutable_data<Tout>(ctx.GetPlace());
    auto axis = ctx.Attr<int64_t>("axis");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

#define CALL_ARG_MINMAX_FUNCTOR(rank)                                \
  ArgMinMaxFunctor<DeviceContext, T, Tout, rank, EnumArgMinMaxValue> \
      functor##rank;                                                 \
  functor##rank(dev_ctx, x, out, axis)

    switch (x.dims().size()) {
      case 1:
        CALL_ARG_MINMAX_FUNCTOR(1);
        break;
      case 2:
        CALL_ARG_MINMAX_FUNCTOR(2);
        break;
      case 3:
        CALL_ARG_MINMAX_FUNCTOR(3);
        break;
      case 4:
        CALL_ARG_MINMAX_FUNCTOR(4);
        break;
      case 5:
        CALL_ARG_MINMAX_FUNCTOR(5);
        break;
      case 6:
        CALL_ARG_MINMAX_FUNCTOR(6);
        break;
      default:
        PADDLE_THROW(
            "%s operator doesn't supports tensors whose ranks are greater "
            "than 6.",
            (EnumArgMinMaxValue == kArgMin ? "argmin" : "argmax"));
        break;
    }
  }
};

template <typename DeviceContext, typename T, typename Tout>
using ArgMinKernel =
    ArgMinMaxKernel<DeviceContext, T, Tout, ArgMinMaxType::kArgMin>;

template <typename DeviceContext, typename T, typename Tout>
using ArgMaxKernel =
    ArgMinMaxKernel<DeviceContext, T, Tout, ArgMinMaxType::kArgMax>;

typedef class BaseArgMinMaxOp : public framework::OperatorWithKernel {
 public:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null");
    const auto& x_dims = ctx->GetInputDim("X");
    int64_t axis = ctx->Attrs().Get<int64_t>("axis");
    PADDLE_ENFORCE(axis >= -x_dims.size() && axis < x_dims.size(),
                   "'axis' must be inside [-Rank(X), Rank(X))");

    auto x_rank = x_dims.size();
    if (axis < 0) axis += x_rank;

    std::vector<int64_t> vec;
    for (int64_t i = 0; i < axis; i++) vec.push_back(x_dims[i]);
    for (int64_t i = axis + 1; i < x_rank; i++) vec.push_back(x_dims[i]);
    ctx->SetOutputDim("Out", framework::make_ddim(vec));
  }
} ArgMinOp, ArgMaxOp;

class BaseArgMinMaxOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  virtual const char* OpName() const = 0;
  virtual const char* Name() const = 0;

 public:
  void Make() override {
    AddInput("X", "Input tensor.");
    AddOutput("Out", "Output tensor.");
    AddAttr<int64_t>("axis", "The axis in which to compute the arg indics.");
    AddComment(::paddle::string::Sprintf(R"DOC(
				%s Operator.

				Computes the indices of the %s elements of the input tensor's element along the provided axis.
)DOC",
                                         OpName(), Name()));
  }
};

class ArgMinOpMaker : public BaseArgMinMaxOpMaker {
 protected:
  const char* OpName() const override { return "ArgMin"; }
  const char* Name() const override { return "min"; }
};

class ArgMaxOpMaker : public BaseArgMinMaxOpMaker {
 protected:
  const char* OpName() const override { return "ArgMax"; }
  const char* Name() const override { return "max"; }
};
}  // namespace operators
}  // namespace paddle

#define REGISTER_ARG_MINMAX_OP_WITHOUT_GRADIENT(op_type, op_name)       \
  REGISTER_OP_WITHOUT_GRADIENT(op_type, paddle::operators::op_name##Op, \
                               paddle::operators::op_name##OpMaker)

#define REGISTER_ARG_MINMAX_KERNEL(op_type, op_name, library_type)           \
  REGISTER_OP_##library_type##_KERNEL(                                       \
      op_type,                                                               \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, float, int64_t>,    \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, double, int64_t>,   \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, int64_t, int64_t>,  \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, int32_t, int64_t>,  \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, int16_t, int64_t>,  \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, int8_t, int64_t>,   \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, uint64_t, int64_t>, \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, uint32_t, int64_t>, \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, uint16_t, int64_t>, \
      paddle::operators::op_name##Kernel<                                    \
          paddle::platform::library_type##DeviceContext, uint8_t, int64_t>)
