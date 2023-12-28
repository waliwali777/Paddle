#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/pir/core/builtin_attribute.h"

namespace cinn::dialect {
using namespace symbol;

namespace {

template <typename T>
std::string GetSerializedTag();

template<>
std::string GetSerializedTag<Negative<DimExpr>>() {
  return "Negative";
}

template<>
std::string GetSerializedTag<Reciprocal<DimExpr>>() {
  return "Reciprocal";
}

template<>
std::string GetSerializedTag<Add<DimExpr>>() {
  return "Add";
}

template<>
std::string GetSerializedTag<Mul<DimExpr>>() {
  return "Mul";
}

template<>
std::string GetSerializedTag<Max<DimExpr>>() {
  return "Max";
}

template<>
std::string GetSerializedTag<Min<DimExpr>>() {
  return "Min";
}

template<>
std::string GetSerializedTag<Broadcast<DimExpr>>() {
  return "Broadcast";
}

template<>
std::string GetSerializedTag<Min<DimExpr>>() {
  return "Min";
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const std::int64_t& dim_expr) {
  return pir::Int64Attribute::get(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const std::string& dim_expr) {
  return pir::StrAttribute::get(ctx, dim_expr);
}

template <typename T>
::pir::Attribute ConvertUnaryDimExprToAttributeImpl(::pir::IrContext* ctx, const T& dim_expr) {
  std::vector<::pir::Attribute> attr_vecs{};
  attr_vecs.push_back(pir::StrAttribute::get(ctx, GetSerializedTag<T>()));
  const auto& [operand] = *dim_expr;
  attr_vecs.push_back(ConvertDimExprToAttribute(ctx, operand));
  return pir::ArrayAttribute::get(ctx, attr_vecs);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Negative<DimExpr>& dim_expr) {
  return ConvertUnaryDimExprToAttributeImpl(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Reciprocal<DimExpr>& dim_expr) {
  return ConvertUnaryDimExprToAttributeImpl(ctx, dim_expr);
}

template<typename T>
::pir::Attribute ConvertVariadicDimExprToAttribute(::pir::IrContext* ctx, const T& dim_expr) {
  std::vector<::pir::Attribute> attr_vecs{};
  attr_vecs.push_back(pir::StrAttribute::get(ctx, GetSerializedTag<T>()));
  const auto& operands = *dim_expr;
  for (const auto& operand : operands) {
    attr_vecs.push_back(ConvertDimExprToAttribute(ctx, operand));
  }
  return pir::ArrayAttribute::get(ctx, attr_vecs);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Add<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Mul<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Max<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Min<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx, const Broadcast<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

std::optional<DimExpr> ConvertInt64AttributeToDimExpr(const ::pir::Int64Attribute& attribute) {
  return DimExpr{attribute.data()};
}


std::optional<DimExpr> ConvertStrAttributeToDimExpr(const ::pir::StrAttribute& attribute) {
  return DimExpr{attribute.AsString()};
}

template <typename T>
std::optional<DimExpr> ConvertArrayAttributeToUnaryDimExpr(const ::pir::ArrayAttribute& attribute) {
  if (attribute.size() != 2) return std::nullopt;
  std::optional<DimExpr> operand = ConvertAttributeToDimExpr(attribute.at(1));
  if (!operand.has_value()) return std::nullopt;
  return T{operand.value()};
}


template <typename T>
std::optional<DimExpr> ConvertArrayAttributeToVariadicDimExpr(const ::pir::ArrayAttribute& attribute) {
  if (attribute.size() < 2) return std::nullopt;
  List<DimExpr> operands{};
  for (std::size_t i = 1; i < attribute.size(); ++i) {
    std::optional<DimExpr> operand = ConvertAttributeToDimExpr(attribute.at(i));
    if (!operand.has_value()) return std::nullopt;
    operands.push_back(operand.value());
  }
  return T{operands};
}

typedef std::optional<DimExpr> (*ArrayAttributeConverterT)(const ::pir::ArrayAttribute& attribute);

std::optional<ArrayAttributeConverterT> GetArrayAttributeConverter(const std::string& tag) {
  static std::unordered_map<std::string, ArrayAttributeConverterT> map{
    {GetSerializedTag<Negative<DimExpr>>(),
      &ConvertArrayAttributeToUnaryDimExpr<Negative<DimExpr>>},
    {GetSerializedTag<Reciprocal<DimExpr>>(),
      &ConvertArrayAttributeToUnaryDimExpr<Reciprocal<DimExpr>>},
    {GetSerializedTag<Add<DimExpr>>(),
      &ConvertArrayAttributeToVariadicDimExpr<Add<DimExpr>>},
    {GetSerializedTag<Mul<DimExpr>>(),
      &ConvertArrayAttributeToVariadicDimExpr<Mul<DimExpr>>},
    {GetSerializedTag<Max<DimExpr>>(),
      &ConvertArrayAttributeToVariadicDimExpr<Max<DimExpr>>},
    {GetSerializedTag<Min<DimExpr>>(),
      &ConvertArrayAttributeToVariadicDimExpr<Min<DimExpr>>},
    {GetSerializedTag<Broadcast<DimExpr>>(),
      &ConvertArrayAttributeToVariadicDimExpr<Broadcast<DimExpr>>},
  };
  const auto& iter = map.find(tag);
  if (iter == map.end()) return std::nullopt;
  return iter->second;
}

std::optional<DimExpr> ConvertArrayAttributeToDimExpr(const ::pir::ArrayAttribute& attribute) {
  if (attribute.empty()) return std::nullopt;
  if (!attribute.at(0).isa<::pir::StrAttribute>()) return std::nullopt;
  const auto& tag = attribute.at(0).dyn_cast<::pir::StrAttribute>().AsString();
  auto opt_func = GetArrayAttributeConverter(tag);
  if (!opt_func.has_value()) return std::nullopt;
  return opt_func.value()(attribute);
}

}  // namespace

::pir::Attribute ConvertDimExprToAttribute(pir::IrContext* ctx, const DimExpr& dim_expr) {
  return std::visit([&](const auto& impl){
    return ConvertDimExprToAttributeImpl(ctx, impl);
  }, dim_expr.variant());
}

std::optional<DimExpr> ConvertAttributeToDimExpr(::pir::Attribute attribute) {
  if (attribute.isa<::pir::Int64Attribute>()) {
    return ConvertInt64AttributeToDimExpr(attribute.dyn_cast<::pir::Int64Attribute>());
  }
  if (attribute.isa<::pir::StrAttribute>()) {
    return ConvertStrAttributeToDimExpr(attribute.dyn_cast<::pir::StrAttribute>());
  }
  if (attribute.isa<::pir::ArrayAttribute>()) {
    return ConvertArrayAttributeToDimExpr(attribute.dyn_cast<::pir::ArrayAttribute>());
  }
  return std::nullopt;
}

namespace {

std::optional<DimExpr> GetDimExprBySymbolBindingImpl(
    const GenerateShapeOp::DataSymbolBinding& symbol_binding,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>& DimExpr4InputDim) {
  const symbol::ShapeOrDataDimExprs& shape_or_data_dim_expr =
    DimExpr4InputDim(symbol_binding.input_tensor_idx);
  if (!shape_or_data_dim_expr.data().has_value()) return std::nullopt;
  int dim_idx = symbol_binding.input_tensor_dim_idx;
  if (dim_idx >= shape_or_data_dim_expr.data().value().size()) return std::nullopt;
  return shape_or_data_dim_expr.data().value().at(dim_idx);
}

std::optional<DimExpr> GetDimExprBySymbolBindingImpl(
    const GenerateShapeOp::ShapeSymbolBinding& symbol_binding,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>& DimExpr4InputDim) {
  const symbol::ShapeOrDataDimExprs& shape_or_data_dim_expr =
    DimExpr4InputDim(symbol_binding.input_tensor_idx);
  int dim_idx = symbol_binding.input_tensor_dim_idx;
  if (dim_idx >= shape_or_data_dim_expr.shape().size()) return std::nullopt;
  return shape_or_data_dim_expr.shape().at(dim_idx);
}

}

std::function<std::optional<DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const GenerateShapeOp::SymbolBindings& symbol_bindings,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>& DimExpr4InputDim) {
  std::unordered_map<std::string, std::vector<GenerateShapeOp::SymbolBinding>> symbol_name2symbol_bindins{};
  const auto& GetDimExpr = [&](const GenerateShapeOp::SymbolBinding& symbol_binding) {
    return std::visit([&](const auto& impl) {
      return GetDimExprBySymbolBindingImpl(impl, DimExpr4InputDim);
    }, symbol_binding);
  };
  return [map=std::move(symbol_name2symbol_bindins),GetDimExpr](const std::string& symbol_name) -> std::optional<DimExpr> {
    const auto& iter = map.find(symbol_name);
    if (iter == map.end()) return std::nullopt;
    std::optional<DimExpr> ret = std::nullopt;
    for (const auto& symbol_binding : iter->second) {
      const auto& current = GetDimExpr(symbol_binding);
      if (!current.has_value()) return std::nullopt;
      if (ret.has_value()) {
        // Same names, same DimExprs.
        if (ret.value() != current.value()) return std::nullopt;
      } else {
        ret = current;
      }
    }
    return ret;
  };
}

}