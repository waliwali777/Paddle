/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(benchmark);
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");
DEFINE_int32(inner_op_parallelism, 0, "number of threads for inner op");

namespace paddle {
namespace framework {

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority = {
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN),
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain),
    std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN),
    std::make_tuple(platform::CPUPlace(), LibraryType::kPlain),
};

proto::VarType::Type GetDataTypeOfVar(const Variable* var) {
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().type();
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().value().type();
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
}

static DDim GetDims(const Scope& scope, const std::string& name,
                    bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return DDim({-1});
    }
    return tensor.dims();
  } else if (var->IsType<SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<SelectedRows>().value().dims();
    } else {
      return var->Get<SelectedRows>().GetCompleteDims();
    }
  } else {
    return DDim({-1});
  }
}

static bool VarInited(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) return false;
  return var->IsInitialized();
}

static std::string GetDtype(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return DataTypeToString(tensor.type());
  } else if (var->IsType<SelectedRows>()) {
    auto tensor = var->Get<SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return DataTypeToString(tensor.type());
    }
  } else {
    return "";
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<SelectedRows>()) {
    return var->Get<SelectedRows>().rows().size();
  }

  return -1;
}

static LoD GetLoD(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return default_lod;
    }
    return tensor.lod();
  } else {
    return default_lod;
  }
}

RuntimeContext::RuntimeContext(const VariableNameMap& innames,
                               const VariableNameMap& outnames,
                               const Scope& scope) {
  for (auto& var_name_item : innames) {
    std::vector<Variable*>& input_vars = inputs[var_name_item.first];
    input_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      input_vars.push_back(scope.FindVar(var_name));
    }
  }
  for (auto& var_name_item : outnames) {
    std::vector<Variable*>& output_vars = outputs[var_name_item.first];
    output_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      output_vars.push_back(scope.FindVar(var_name));
    }
  }
}

void OperatorBase::Run(const Scope& scope, const platform::Place& place) {
  try {
    VLOG(4) << place << " " << DebugStringEx(&scope);
    if (platform::is_gpu_place(place)) {
#ifndef PADDLE_WITH_CUDA
      PADDLE_THROW("Cannot run operator on place %s", place);
#else
      auto dev_id = boost::get<platform::CUDAPlace>(place).device;
      platform::SetDeviceId(dev_id);
#endif
    }

    // The profile has a process-wide mutex, results in serious performance
    // issue
    // in concurrency scenerio. Here use an `if` to fix this issue.
    // Please not remove the `if`, ask @Superjomn if there are any concern.
    if (platform::IsProfileEnabled()) {
      platform::RecordEvent record_event(Type());
      RunImpl(scope, place);
    } else {
      RunImpl(scope, place);
    }

    VLOG(3) << place << " " << DebugStringEx(&scope);
  } catch (platform::EnforceNotMet exception) {
    if (Attrs().count("sub_block") != 0) {
      throw;
    }

    auto& callstack = Attr<std::vector<std::string>>(
        OpProtoAndCheckerMaker::OpCreationCallstackAttrName());

    if (callstack.empty()) {
      throw;
    }
    std::ostringstream sout;
    sout << "Invoke operator " << Type() << " error.\n";
    sout << "Python Callstacks: \n";
    for (auto& line : callstack) {
      sout << line;
    }
    sout << "C++ Callstacks: \n";
    sout << exception.err_str_;
    exception.err_str_ = sout.str();
    throw;
  } catch (...) {
    std::rethrow_exception(std::current_exception());
  }
}

bool OperatorBase::HasInputs(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_, name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
                 type_, name);
  return it->second;
}

bool OperatorBase::HasOutputs(const std::string& name) const {
  if (outputs_.find(name) != outputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_, name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.", type_, name);
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto var_name = input.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, var_name);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, var_name);
          ss << ":" << dtype;
          ss << "[" << GetDims(*scope, var_name, true) << "]";
          ss << "(" << GetLoD(*scope, var_name) << ")";
        }
      }
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto var_name = output.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, output.second[i]);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, output.second[i]);
          ss << ":" << dtype;
          ss << "[" << GetDims(*scope, var_name, true) << "]";
          ss << "(" << GetLoD(*scope, var_name) << ")";
        }
      }
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  GenerateTemporaryNames();
  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = OpInfoMap::Instance().Get(Type());

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  auto& info_map = OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) return;

  for (auto& in : op_info->Proto().inputs()) {
    if (!in.dispensable()) {
      PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
                     "Operator %s's input, %s, is not set", Type(), in.name());
    }
  }

  for (auto& out : op_info->Proto().outputs()) {
    if (!out.dispensable()) {
      PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
                     "Operator %s's output, %s, is not set", Type(),
                     out.name());
    }
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static bool VarIsTensor(const Variable& var) {
  return var.IsType<LoDTensor>() || var.IsType<SelectedRows>();
}

const Tensor* GetLoDTensorOrSelectedRowsValueFromVar(const Variable& var) {
  if (var.IsType<LoDTensor>()) {
    return static_cast<const Tensor*>(&(var.Get<LoDTensor>()));
  } else if (var.IsType<SelectedRows>()) {
    return &(var.Get<SelectedRows>().value());
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 ToTypeName(var.Type()));
  }
}

Tensor* GetMutableLoDTensorOrSelectedRowsValueFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 ToTypeName(var->Type()));
  }
}

bool ExecutionContext::HasInput(const std::string& name) const {
  if (!op_.HasInputs(name)) {
    return false;
  }
  auto& ins = Inputs(name);
  size_t length = ins.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input %s should not have more than one inputs", name);
  auto arg = ins[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  if (!op_.HasOutputs(name)) {
    return false;
  }
  auto& outs = Outputs(name);
  size_t length = outs.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Output %s should not have more than one inputs", name);
  auto arg = outs[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

const Variable* ExecutionContext::InputVar(const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(it->second.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    op_.Type(), name);
  return it->second.empty() ? nullptr : it->second[0];
}

Variable* ExecutionContext::OutputVar(const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(it->second.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    op_.Type(), name);
  return it->second.empty() ? nullptr : it->second[0];
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  return Input<LoDTensor>(name);
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) {
    return {};
  }
  const std::vector<Variable*>& vars = it->second;
  std::vector<const Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](Variable* var) -> const Tensor* {
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE(
                       var->IsType<LoDTensor>(),
                       "should be LoDTensor, but the received type is %s",
                       ToTypeName(var->Type()));
                   return &(var->Get<LoDTensor>());
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  return Output<LoDTensor>(name);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) {
    return {};
  }
  const std::vector<Variable*>& vars = it->second;
  std::vector<Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](Variable* var) -> Tensor* {
                   return var == nullptr ? nullptr
                                         : var->GetMutable<LoDTensor>();
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

static void CheckTensorNANOrInf(const std::string& op_type,
                                const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type() != proto::VarType::FP32 &&
      tensor.type() != proto::VarType::FP64) {
    return;
  }
  PADDLE_ENFORCE(!framework::TensorContainsInf(tensor),
                 "Operator %s output Tensor %s contains Inf", op_type, name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(tensor),
                 "Operator %s output Tensor %s contains NAN", op_type, name);
}

void OperatorWithKernel::RuntimeInferShape(const Scope& scope,
                                           const platform::Place& place,
                                           const RuntimeContext& ctx) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope, ctx);
  this->InferShape(&infer_shape_ctx);
}

std::vector<KernelConfig>* OperatorWithKernel::GetKernelConfig(
    const OpKernelType& key) const {
  auto config_iter = kernel_configs_map_.find(key);
  std::vector<KernelConfig>* kernel_configs = nullptr;
  if (config_iter != kernel_configs_map_.end()) {
    kernel_configs = &(config_iter->second);
  }
  return kernel_configs;
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place) const {
  RuntimeContext ctx(Inputs(), Outputs(), scope);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  OpKernelMap& kernels = kernels_iter->second;

  auto expected_kernel_key = this->GetExpectedKernelType(
      ExecutionContext(*this, scope, *dev_ctx, ctx, nullptr));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
  // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
  if (kernel_iter == kernels.end() &&
      expected_kernel_key.library_type_ == LibraryType::kMKLDNN) {
    VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
    expected_kernel_key.library_type_ = LibraryType::kPlain;
    expected_kernel_key.data_layout_ = DataLayout::kAnyLayout;
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s", type_,
                 KernelTypeToString(expected_kernel_key));
  }

  std::vector<KernelConfig>* kernel_configs =
      GetKernelConfig(expected_kernel_key);

  // do data transformScope &transfer_scope;
  std::vector<std::string> transfered_inplace_vars;
  auto* transfer_scope =
      PrepareData(scope, expected_kernel_key, &transfered_inplace_vars, &ctx);

  // exec scope is the scope that kernel actually executed on.
  const Scope& exec_scope =
      (transfer_scope == nullptr ? scope : *transfer_scope);

  if (!(expected_kernel_key.place_ == dev_ctx->GetPlace())) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
  }

  RuntimeInferShapeContext infer_shape_ctx(*this, exec_scope, ctx);
  this->InferShape(&infer_shape_ctx);
  // TODO(panyx0718): ExecutionContext should only depend on RuntimeContext
  // not Scope. Imperative mode only pass inputs and get outputs.
  kernel_iter->second(
      ExecutionContext(*this, exec_scope, *dev_ctx, ctx, kernel_configs));

  if (!transfered_inplace_vars.empty()) {
    // there is inplace variable has been transfered.
    TransferInplaceVarsBack(scope, transfered_inplace_vars, *transfer_scope);
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    dev_ctx->Wait();
  }

  if (FLAGS_check_nan_inf) {
    for (auto& vname : OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      if (var->IsType<framework::LoDTensor>()) {
        CheckTensorNANOrInf(type_, vname, var->Get<framework::LoDTensor>());
      } else if (var->IsType<framework::SelectedRows>()) {
        CheckTensorNANOrInf(type_, vname,
                            var->Get<framework::SelectedRows>().value());
      }
    }
  }
}

void OperatorWithKernel::TransferInplaceVarsBack(
    const Scope& scope, const std::vector<std::string>& inplace_vars,
    const Scope& transfer_scope) const {
  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* origin_var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(origin_var, "The var[%s] should not be nullptr.",
                            var_name);
    auto* original_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(origin_var);
    auto* var = transfer_scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(var, "The var[%s] should not be nullptr.",
                            var_name);
    auto* transformed_tensor = GetLoDTensorOrSelectedRowsValueFromVar(*var);
    original_tensor->ShareDataWith(*transformed_tensor);
  }
}

Scope* OperatorWithKernel::PrepareData(
    const Scope& scope, const OpKernelType& expected_kernel_key,
    std::vector<std::string>* transfered_inplace_vars,
    RuntimeContext* ctx) const {
  Scope* new_scope = nullptr;
  for (auto& var_name_item : Inputs()) {
    std::vector<Variable*>& input_vars = ctx->inputs[var_name_item.first];

    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      auto& var_name = var_name_item.second[i];
      auto* var = input_vars[i];

      // Only tensor can be tranfer to another device.
      if (var == nullptr || !VarIsTensor(*var)) {
        continue;
      }

      auto* tensor_in = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      if (!tensor_in->IsInitialized()) {
        continue;
      }

      auto kernel_type_for_var = GetKernelTypeForVar(
          var_name_item.first, *tensor_in, expected_kernel_key);

      if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
        continue;
      }

      auto out_var_names = OutputVars(true);
      if (std::find(out_var_names.begin(), out_var_names.end(), var_name) !=
          out_var_names.end()) {
        transfered_inplace_vars->emplace_back(var_name);
      }

      VLOG(3) << "Transform Variable " << var_name << " from "
              << kernel_type_for_var << " to " << expected_kernel_key;

      // In the inference scenerio, the scopes will be reused across the
      // batches, so the `new_scope` here will result in GPU memroy explosion
      // over the  running of operators.
      // We use a thread_local cache to fix that issue, the key in the cache is
      // the combination of the `scope` argument, from_kernel_type,
      // target_kernel_type.
      // Have a discussion with @Superjomn or the inference developers if some
      // changes on this logic for this macro might not tested on the other
      // scenerios.
      // If this op is not called by an Executor or ParallelExecutor, it should
      // called by a NaiveExecutor, the NaiveExecutor will cache the scopes and
      // variables, that behavior a lot different.
      if (!run_by_executor_) {
        new_scope = TryCreateTransferScope(kernel_type_for_var,
                                           expected_kernel_key, &scope);
      }
      if (!new_scope) {
        new_scope = &scope.NewScope();
      }

      auto* trans_var = new_scope->Var(var_name);
      input_vars[i] = trans_var;

      Tensor out;
      TransformData(expected_kernel_key, kernel_type_for_var, *tensor_in, &out);
      SetTensorToVariable(*var, out, trans_var);
    }
  }

  return new_scope;
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  proto::VarType::Type dafault_data_type =
      static_cast<proto::VarType::Type>(-1);
  proto::VarType::Type data_type = dafault_data_type;
  for (auto& input : this->inputs_) {
    const std::vector<const Variable*> vars = ctx.MultiInputVar(input.first);
    for (size_t i = 0; i < vars.size(); ++i) {
      const Variable* var = vars[i];
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          PADDLE_ENFORCE(t->IsInitialized(), "Input %s(%lu)is not initialized",
                         input.first, i);
          proto::VarType::Type tmp = t->type();
          PADDLE_ENFORCE(
              tmp == data_type || data_type == dafault_data_type,
              "DataType of Paddle Op %s must be the same. Get (%d) != (%d)",
              Type(), DataTypeToString(data_type), DataTypeToString(tmp));
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != dafault_data_type,
                 "DataType should be indicated by input");
  return data_type;
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const OpKernelType& expected_kernel_type) const {
  return OpKernelType(expected_kernel_type.data_type_, tensor.place(),
                      tensor.layout());
}

}  // namespace framework
}  // namespace paddle
