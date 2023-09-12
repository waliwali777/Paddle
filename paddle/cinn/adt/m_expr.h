// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <functional>

#include "paddle/cinn/adt/adapter.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/schedule_policy.h"

namespace cinn {
namespace hlir {
namespace framework {
class Node;
class NodeData;
}  // namespace framework
}  // namespace hlir
}  // namespace cinn

namespace cinn {
namespace adt {
namespace m_expr {

// Offset = Int64
using Offset = std::int64_t;

class GlobalMemoryType final {};

inline std::size_t GetHashValue(const GlobalMemoryType&) {
  static GlobalMemoryType global_memory_type;
  return &global_memory_type;
}

class SharedMemoryType final {};

inline std::size_t GetHashValue(const SharedMemoryType&) {
  static SharedMemoryType shared_memory_type;
  return &shared_memory_type;
}

// MemoryType = GlobalMemoryType | SharedMemoryType
DEFINE_ADT_UNION(MemoryType, GlobalMemoryType, SharedMemoryType);
OVERRIDE_UNION_GET_HASH_VALUE(MemoryType);

// TempStorage = (tVar Name, Offset, MemoryType)
class TempStorage final : public Tuple<tVar<Name>, Offset, MemoryType> {
 public:
  using Tuple<tVar<Name>, Offset, MemoryType>::Tuple;
};
OVERLOAD_OPERATOR_EQ_NE(TempStorage, TupleEqual);
inline std::size_t GetHashValue(const TempStorage& temp_storage) {
  const auto& [var_name, offset, memory_type] = temp_storage.tuple();
  std::size_t hash_value = GetHashValue(var_name.value());
  hash_value = hash_combine(hash_value, offset);
  hash_value = hash_combine(hash_value, GetHashValue(memory_type));
  return hash_value;
}

// SSAShadowTensor = (tSSAShadow Name, const Graph::NodeData*)
class SSAShadowTensor final : public Tuple<tSSAShadow<Name>, m_expr::Tensor> {
 public:
  using Tuple<tSSAShadow<Name>, m_expr::Tensor>::Tuple;
};
OVERLOAD_OPERATOR_EQ_NE(tSSAShadow<Name>, TagEqual);
OVERLOAD_OPERATOR_EQ_NE(SSAShadowTensor, TupleEqual);

OVERRIDE_TAG_GET_HASH_VALUE(tSSAShadow<Name>);

inline std::size_t GetHashValue(const SSAShadowTensor& shadow_tensor) {
  const auto& [shadow_name, tensor] = shadow_tensor.tuple();
  return hash_combine(GetHashValue(shadow_name), tensor);
}

// Tensor = adapter::Tensor | SSAShadowTensor | TempStorage
DEFINE_ADT_UNION(Tensor, adapter::Tensor, SSAShadowTensor, TempStorage);
OVERRIDE_UNION_GET_HASH_VALUE(Tensor);
OVERLOAD_OPERATOR_EQ_NE(Tensor, UnionEqual);

// MemoryBarrier = {}    // (Sync Thread)
class MemoryBarrier final {};

// BuiltinReduceRelatedOp = Zeros | InplaceAdd
class Zeros final {};
class InplaceAdd final {};
DEFINE_ADT_UNION(BuiltinReduceRelatedOp, Zeros, InplaceAdd);

// Op = const Node* | BuiltinReduceRelatedOp | MemoryBarrier
DEFINE_ADT_UNION(Op,
                 const hlir::framework::Node*,
                 BuiltinReduceRelatedOp,
                 MemoryBarrier);

using Arg = Tensor;

// OpStmt = (Op, In [Arg], Out [Arg])
class OpStmt final : public Tuple<Op, tIn<List<Arg>>, tOut<List<Arg>>> {
 public:
  using Tuple<Op, tIn<List<Arg>>, tOut<List<Arg>>>::Tuple;

  bool operator==(const OpStmt& other) const {
    return &this->tuple() == &other.tuple();
  }
};

inline std::size_t GetHashValue(const OpStmt& op_stmt_node) {
  return &op_stmt_node.tuple();
}

// MapStmt T = (ScheduleDescriptor, [T])
template <typename T>
class MapStmt final : public Tuple<ScheduleDescriptor, List<T>> {
 public:
  using Tuple<ScheduleDescriptor, List<T>>::Tuple;
};

// Stmt = OpStmt | MapStmt Stmt
DEFINE_ADT_UNION(Stmt, OpStmt, MapStmt<Stmt>);

using TensorIndexExpr = equation::Value;
using TensorIndexExpr4TensorT =
    std::function<const TensorIndexExpr&(const Tensor&)>;

// AnchoredMapStmt = (MapStmt Stmt, tAnchor Tensor, TensorIndexExpr4TensorT)
class AnchoredMapStmt final
    : public Tuple<MapStmt<Stmt>, tAnchor<Tensor>, TensorIndexExpr4TensorT> {
 public:
  using Tuple<MapStmt<Stmt>, tAnchor<Tensor>, TensorIndexExpr4TensorT>::Tuple;
};

// Kernel = ([AnchoredMapStmt], In [Tensor], Out [Tensor])
class Kernel final : public Tuple<List<AnchoredMapStmt>,
                                  tIn<List<Tensor>>,
                                  tOut<List<Tensor>>> {
 public:
  using Tuple<List<AnchoredMapStmt>, tIn<List<Tensor>>, tOut<List<Tensor>>>::
      Tuple;
};

// MapExpr = Kernel;
using MapExpr = Kernel;

}  // namespace m_expr
}  // namespace adt
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::adt::m_expr::Tensor> {
  std::size_t operator()(const cinn::adt::m_expr::Tensor& tensor) const {
    return cinn::adt::m_expr::GetHashValue(tensor);
  }
};

template <>
struct hash<cinn::adt::m_expr::OpStmt> {
  std::size_t operator()(const cinn::adt::m_expr::OpStmt& op_stmt_node) const {
    return cinn::adt::m_expr::GetHashValue(op_stmt_node);
  }
};

}  // namespace std
