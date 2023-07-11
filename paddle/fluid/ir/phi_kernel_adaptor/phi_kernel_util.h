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

#pragma once

#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"

#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/phi/core/infermeta_utils.h"

#include "glog/logging.h"

namespace ir {
paddle::framework::Variable* CreateVar(ir::Value value,
                                       const std::string& name,
                                       paddle::framework::Scope* scope,
                                       paddle::framework::Scope* local_scope);

void BuildValue(ir::Value value,
                paddle::framework::Scope* scope,
                paddle::framework::Scope* local_scope,
                std::unordered_map<ir::Value, std::string>* name_map,
                int& count);  // NOLINT

void HandleForSpecialOp(ir::Operation* op,
                        paddle::framework::Scope* scope,
                        paddle::framework::Scope* local_scope,
                        std::unordered_map<ir::Value, std::string>* name_map,
                        int& count);  // NOLINT

void HandleForInplaceOp(ir::Operation* op,
                        paddle::framework::Scope* scope,
                        paddle::framework::Scope* local_scope,
                        std::unordered_map<ir::Value, std::string>* name_map,
                        int& count);  // NOLINT

void CheckInputVars(ir::Operation* op,
                    const std::unordered_map<ir::Value, std::string>& name_map);

void BuildScope(const ir::Block& block,
                paddle::framework::Scope* scope,
                paddle::framework::Scope* local_scope,
                std::unordered_map<ir::Value, std::string>* name_map);

template <typename Context,
          typename InType,
          typename OutType,
          typename ListType,
          bool is_kernel>
void BuildPhiContext(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    paddle::framework::Scope* local_scope,
    const paddle::dialect::OpYamlInfoParser& op_yaml_info,
    Context* ctx,
    std::map<std::string, std::vector<int>>* input_map = nullptr,
    std::map<std::string, std::vector<int>>* output_map = nullptr) {
  paddle::framework::Scope* inner_scope =
      local_scope != nullptr ? local_scope : scope;
  VLOG(6) << "BuildPhiContext in scope[" << scope << "] inner_scope["
          << inner_scope << "]";
  // inputs include input and mutable attributes

  auto attr_map = op->attributes();

  auto& vec_kernel_fn_tensor_params = op_yaml_info.TensorParams(is_kernel);

  auto& name2id = op_yaml_info.InputName2Id();
  for (auto& t : vec_kernel_fn_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(t),
        true,
        phi::errors::NotFound("param [%s] MUST in name2id map", t));
    auto index = op_yaml_info.InputName2Id().at(t);
    ir::Value ptr = op->operand(index);
    if (!ptr) {
      phi::DenseTensor* ptr = nullptr;
      OutType in_ptr(ptr);
      ctx->EmplaceBackInput(in_ptr);
      continue;
    }

    auto in_var_name = name_map.at(ptr);
    VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            phi::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    if (var->IsType<phi::DenseTensor>()) {
      const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
      ctx->EmplaceBackInput(InType(tensor_in));
    } else if (var->IsType<paddle::framework::TensorRefArray>()) {
      ListType inputs;
      auto& tensor_array = var->Get<paddle::framework::TensorRefArray>();
      for (size_t i = 0; i < tensor_array.size(); ++i) {
        inputs.emplace_back(InType(tensor_array[i]));
      }
      ctx->EmplaceBackInputs(inputs);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Not support var type [%d] ",
                                              var->Type()));
    }
  }

  auto& vec_kernel_fn_attr_params = op_yaml_info.AttrParams(is_kernel);
  for (auto& t : vec_kernel_fn_attr_params) {
    if (name2id.count(t)) {
      // tensor attribute, get information from input
      ir::Value ptr = op->operand(name2id.at(t));

      auto in_var_name = name_map.at(ptr);
      if (input_map != nullptr) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        // TODO(phlrain): OpFuncNode need input_index and output_index,
        // construct input_index and output_here,  should remove input_index and
        // output_index from OpFuncNode Each in_var_name named "inner_var_" +
        // index, len("inner_var_") = 10

        size_t tmp_id = std::atol(in_var_name.substr(4, 100).c_str());
        (*input_map)[std::to_string(name2id.at(t))].push_back(tmp_id);
      }

      auto& tensor_attr_type = op_yaml_info.TensorAttrTypeName(t);
      VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t" << in_var_name;
      if (tensor_attr_type == "paddle::dialect::IntArrayAttribute") {
        if (ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
          phi::Attribute r1 = phi::TensorRef(
              &(inner_scope->FindVar(in_var_name)->Get<phi::DenseTensor>()));
          ctx->EmplaceBackAttr(r1);
        } else if (ptr.type().isa<ir::VectorType>()) {
          auto& tensor_array = inner_scope->FindVar(in_var_name)
                                   ->Get<paddle::framework::TensorRefArray>();
          if (tensor_array.size() == 1) {
            ctx->EmplaceBackAttr(phi::TensorRef(tensor_array[0]));
          } else {
            std::vector<phi::TensorRef> vec_ref;
            for (size_t i = 0; i < tensor_array.size(); ++i) {
              vec_ref.emplace_back(phi::TensorRef(tensor_array[i]));
            }
            ctx->EmplaceBackAttr(vec_ref);
          }
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              " [%s] only support dense tensor and vector type  ",
              tensor_attr_type));
        }
      } else if (tensor_attr_type == "paddle::dialect::ScalarAttribute") {
        phi::Attribute attr = phi::TensorRef(
            &(inner_scope->FindVar(in_var_name)->Get<phi::DenseTensor>()));

        ctx->EmplaceBackAttr(attr);
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                tensor_attr_type));
      }

      continue;
    }

    auto& attr_type_name = op_yaml_info.AttrTypeName(t);
    if (attr_type_name == "paddle::dialect::IntArrayAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::DataTypeAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
    } else if (attr_type_name == "ir::Int32Attribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::Int32Attribute>().data());
    } else if (attr_type_name == "ir::Int64Attribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::Int64Attribute>().data());
    } else if (attr_type_name == "ir::FloatAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::FloatAttribute>().data());
    } else if (attr_type_name == "ir::BoolAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::BoolAttribute>().data());
    } else if (attr_type_name == "ir::StrAttribute") {
      ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::StrAttribute>().data());
    } else if (attr_type_name ==
               "ir::ArrayAttribute<paddle::dialect::ScalarAttribute>") {
      auto array_list = attr_map[t].dyn_cast<ir::ArrayAttribute>().data();
      std::vector<phi::Scalar> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<paddle::dialect::ScalarAttribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be dialect::ScalarAttribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(array_list[i]
                                .dyn_cast<paddle::dialect::ScalarAttribute>()
                                .data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "ir::ArrayAttribute<ir::Int32Attribute>") {
      auto array_list = attr_map[t].dyn_cast<ir::ArrayAttribute>().data();
      std::vector<int32_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<ir::Int32Attribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be ir::Int32Attribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<ir::Int32Attribute>().data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "ir::ArrayAttribute<ir::FloatAttribute>") {
      auto array_list = attr_map[t].dyn_cast<ir::ArrayAttribute>().data();
      std::vector<float> vec_res;
      if (array_list.size() > 0) {
        if (array_list[0].isa<ir::FloatAttribute>()) {
          for (size_t i = 0; i < array_list.size(); ++i) {
            vec_res.push_back(
                array_list[i].dyn_cast<ir::FloatAttribute>().data());
          }

        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  attr_type_name));
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "ir::ArrayAttribute<ir::Int64Attribute>") {
      auto array_list = attr_map[t].dyn_cast<ir::ArrayAttribute>().data();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<ir::Int64Attribute>(),
            true,
            phi::errors::PreconditionNotMet(
                "Element in array list MUST be ir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<ir::Int64Attribute>().data());
        }
      }
      ctx->EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "paddle::dialect::PlaceAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
    } else if (attr_type_name == "paddle::dialect::ScalarAttribute") {
      ctx->EmplaceBackAttr(
          attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                              attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }

  // TODO(phlrain): use var type instead of op name
  if (op->attributes().count("op_name") &&
      (op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data() ==
       "pd.fetch")) {
    // process fetch op
    auto fetch_var = inner_scope->FindVar("fetch");
    auto* fetch_list = fetch_var->GetMutable<paddle::framework::FetchList>();
    int index =
        op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
    auto* out_tensor = &(PADDLE_GET(phi::DenseTensor, fetch_list->at(index)));
    ctx->EmplaceBackOutput(out_tensor);
  } else {
    for (size_t i = 0; i < op->num_results(); ++i) {
      ir::Value out_ptr = op->result(i);
      auto name = name_map.at(out_ptr);
      if (out_ptr.type()) {
        ctx->EmplaceBackOutput(OutType(const_cast<phi::DenseTensor*>(
            &(inner_scope->FindVar(name)->Get<phi::DenseTensor>()))));
      } else {
        phi::DenseTensor* ptr = nullptr;
        OutType out_ptr(ptr);
        ctx->EmplaceBackOutput(out_ptr);
      }

      if (output_map != nullptr) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        // TODO(phlrain): OpFuncNode need input_index and output_index,
        // construct input_index and output_here,  should remove input_index and
        // output_index from OpFuncNode Each in_var_name named "inner_var_" +
        // index, len("inner_var_") = 10

        size_t tmp_id = std::atol(name.substr(4, 100).c_str());
        (*output_map)["out"].push_back(tmp_id);
      }
    }
  }
}

}  // namespace ir
