// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
// NOTE(zhangbo9674): File pd_op.h is generated by op_gen.py, see details in
// paddle/fluid/ir/dialect/CMakeLists.txt.
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type_storage.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/transforms/param_to_variable.h"
#include "paddle/ir/core/ir_printer.h"
#include "paddle/ir/core/utils.h"

namespace paddle {
namespace dialect {

PaddleDialect::PaddleDialect(ir::IrContext *context)
    : ir::Dialect(name(), context, ir::TypeId::get<PaddleDialect>()) {
  initialize();
}

void PaddleDialect::initialize() {
  RegisterTypes<paddle::dialect::DenseTensorType>();
  RegisterTypes<paddle::dialect::SelectedRowsType>();

  RegisterAttributes<paddle::dialect::IntArrayAttribute,
                     paddle::dialect::DataTypeAttribute,
                     paddle::dialect::PlaceAttribute,
                     paddle::dialect::DataLayoutAttribute>();

  // NOTE(zhangbo9674): GET_OP_LIST is defined in pd_op.h which is
  // generated by op_gen.py, see details in
  // paddle/fluid/ir/dialect/CMakeLists.txt.
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"  // NOLINT
      >();
  RegisterOps<
#define GET_MANUAL_OP_LIST
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_op.h"  // NOLINT
      >();

  RegisterInterfaces<ParameterConvertInterface>();
}

void PaddleDialect::PrintType(ir::Type type, std::ostream &os) const {
  os << type.dialect().name();
  os << '.';
  if (auto tensor_type = type.dyn_cast<DenseTensorType>()) {
    os << "tensor<";
    for (auto d : phi::vectorize(tensor_type.dims())) {
      os << d;
      os << "x";
    }
    tensor_type.dtype().Print(os);
    os << ">";
  } else if (auto selected_rows_type = type.dyn_cast<SelectedRowsType>()) {
    os << "selectedrows<";
    for (auto d : phi::vectorize(selected_rows_type.dims())) {
      os << d;
      os << "x";
    }
    selected_rows_type.dtype().Print(os);
    os << ">";
  }
}

void PaddleDialect::PrintAttribute(ir::Attribute attr, std::ostream &os) const {
  if (auto int_array_attr = attr.dyn_cast<IntArrayAttribute>()) {
    phi::IntArray data = int_array_attr.data();
    os << "IntArray[";
    const auto &inner_data = data.GetData();
    ir::PrintInterleave(
        inner_data.begin(),
        inner_data.end(),
        [&os](int64_t i) { os << i; },
        [&os]() { os << ","; });
    os << "]";
  } else if (auto data_type_attr = attr.dyn_cast<DataTypeAttribute>()) {
    os << data_type_attr.data();
  } else if (auto place_type_attr = attr.dyn_cast<PlaceAttribute>()) {
    os << place_type_attr.data();
  } else if (auto data_layout_attr = attr.dyn_cast<DataLayoutAttribute>()) {
    os << data_layout_attr.data();
  } else {
    os << "<#AttrNotImplemented>";
  }
}

void PaddleDialect::PrintOperation(ir::Operation *op,
                                   ir::IrPrinter &printer) const {
  if (auto if_op = op->dyn_cast<IfOp>()) {
    if_op.Print(printer);
  } else {
    printer.PrintGeneralOperation(op);
  }
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PaddleDialect)
