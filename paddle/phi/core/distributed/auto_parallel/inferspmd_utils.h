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

#pragma once

#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/any.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace phi {
namespace distributed {

class InferSpmdContext {
 public:
  InferSpmdContext() = default;

  void EmplaceBackInput(MetaTensor input);
  void EmplaceBackAttr(Attribute attr);

  const MetaTensor& InputAt(size_t idx) const;

  template <typename AttrType>
  const AttrType& AttrAt(size_t idx) const;

  const Attribute& AttrAt(size_t idx) const;

 private:
  paddle::small_vector<MetaTensor, phi::kInputSmallVectorSize> inputs_;
  paddle::small_vector<Attribute, phi::kAttrSmallVectorSize> attrs_;
};

using InferSpmdFn = SpmdInfo (*)(const InferSpmdContext&);

#define PD_INFER_SPMD(...)                                    \
  ::phi::distributed::InferSpmdFnImpl<decltype(&__VA_ARGS__), \
                                      &__VA_ARGS__>::Call

template <typename T>
struct InferSpmdTypeTag {};

template <typename Fn, Fn fn>
struct InferSpmdFnImpl;

template <typename Return, typename... Args, Return (*infer_spmd_fn)(Args...)>
struct InferSpmdFnImpl<Return (*)(Args...), infer_spmd_fn> {
  static SpmdInfo Call(const InferSpmdContext& ctx) {
    return InferSpmdFnCallHelper<Args..., InferSpmdTypeTag<int>>::
        template Call<0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct InferSpmdFnCallHelper;

  // TODO(chenweihang): support other input type later
  template <typename... Tail>
  struct InferSpmdFnCallHelper<const MetaTensor&, Tail...> {
    template <int in_idx, int attr_idx, typename... PreviousArgs>
    static SpmdInfo Call(const InferSpmdContext& ctx, PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "InferSpmd's Input should appear before Attributes.");
      const MetaTensor& arg = ctx.InputAt(in_idx);
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx + 1,
                                                           attr_idx>(
          ctx, pargs..., arg);
    }
  };

#define PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(attr_type)      \
  template <typename... Tail>                                             \
  struct InferSpmdFnCallHelper<attr_type, Tail...> {                      \
    template <int in_idx, int attr_idx, typename... PreviousArgs>         \
    static SpmdInfo Call(const InferSpmdContext& ctx,                     \
                         PreviousArgs&... pargs) {                        \
      attr_type arg = ctx.AttrAt<attr_type>(attr_idx);                    \
      return InferSpmdFnCallHelper<Tail...>::template Call<in_idx,        \
                                                           attr_idx + 1>( \
          ctx, pargs..., arg);                                            \
    }                                                                     \
  }

  // TODO(chenweihang): support other attr type later
  PD_SPECIALIZE_InferSpmdFnCallHelper_FOR_ATTRIBUTE(bool);

  /* End case */
  template <typename T>
  struct InferSpmdFnCallHelper<InferSpmdTypeTag<T>> {
    template <int in_idx, int attr_idx>
    static SpmdInfo Call(const InferSpmdContext& ctx UNUSED, Args&... args) {
      return infer_spmd_fn(args...);
    }
  };
};

// SpmdRuleFactory manage the spmd rules and cache the propagate results
// TODO(chenweihang): Add spmd caching impl later
class SpmdRuleFactory {
 public:
  static SpmdRuleFactory& Instance();

  bool ContainsInferSpmdFn(const std::string& kernel_name) const;

  void InsertInferSpmdFn(std::string kernel_name, InferSpmdFn fn);

  const InferSpmdFn& GetInferSpmdFn(const std::string& kernel_name) const;

 private:
  SpmdRuleFactory() = default;

  paddle::flat_hash_map<std::string, InferSpmdFn> infer_spmd_fn_map_;

  DISABLE_COPY_AND_ASSIGN(SpmdRuleFactory);
};

struct InferSpmdFnRegistrar {
  InferSpmdFnRegistrar(const char* kernel_name, InferSpmdFn fn) {
    SpmdRuleFactory::Instance().InsertInferSpmdFn(kernel_name, fn);
  }
};

#define PD_REGISTER_INFER_SPMD_FN(kernel_name, variadic_infer_spmd_fn)  \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                    \
      PD_REGISTER_infer_spmd_fn_ns_check_##kernel_name,                 \
      "PD_REGISTER_INFER_SPMD_FN must be called in global namespace."); \
  static const ::phi::distributed::InferSpmdFnRegistrar                 \
      __registrar_infer_spmd_fn_for_##kernel_name(                      \
          #kernel_name, PD_INFER_SPMD(variadic_infer_spmd_fn))

}  // namespace distributed
}  // namespace phi
