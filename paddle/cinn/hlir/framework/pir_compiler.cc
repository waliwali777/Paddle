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

#include "paddle/cinn/hlir/framework/pir_compiler.h"

#include <absl/types/variant.h>
#include "paddle/cinn/hlir/framework/pir/compilation_task.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/builtin_type.h"

PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace hlir {
namespace framework {

// TODO(Aurelius84): Need abstract this logic to implement Proxy for
// the co-existance with GraphCompiler.
std::unique_ptr<Program> PirCompiler::Build() {
  m_builder_.Clear();
  // NOTE(Aurelius84): Currently only support each op for one group
  std::vector<pir::GroupPtr> groups;
  for (auto& op : *program_.block()) {
    std::vector<::pir::Operation*> ops = {&op};
    groups.push_back(std::make_shared<pir::Group>(ops));
  }
  VLOG(4) << "Groups size: " << groups.size();
  return std::move(Build(groups));
}

std::vector<pir::CINNKernelInfo> PirCompiler::BuildCUDAJITInfo(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<pir::CINNKernelInfo> cinn_kernel_info_vecs(groups.size());
  if (FLAGS_cinn_bucket_compile) {
    for (int i = 0; i < groups.size(); ++i) {
      group_compilation_contexts_.emplace_back(target_, groups[i], scope_);
    }
    auto worker_fn = [&](int index) {
      CompilationTask task(&group_compilation_contexts_[index]);
      task();
      VLOG(-1) << "zzzz ";
      cinn_kernel_info_vecs[index] = task.BuildPirCINNKernelInfo();
      VLOG(-1) << "zzzz ";
    };
    utils::parallel_run(
        worker_fn, utils::SequenceDispatcher(0, groups.size()), -1);
    VLOG(-1) << "parallel build over";
  } else {
    auto op_lowerer = CreateOpLowerer<pir::GroupPtr>(target_);

    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
    for (int i = 0; i < groups.size(); ++i) {
      lowered_funcs.emplace_back(
          op_lowerer.Lower(groups[i], false, false, false));
    }

    for (auto&& lowered_func : lowered_funcs) {
      ProcessFunction(lowered_func);
    }
    compiler_ = backends::Compiler::Create(target_);
    auto build_module = m_builder_.Build();
    compiler_->Build(build_module, "");

    auto instructions = BuildInstructions(groups);

    auto fn_ptrs = compiler_->GetFnPtr();

    for (int idx = 0; idx < groups.size(); ++idx) {
      pir::CINNKernelInfo cinn_kernel_info;
      auto fn_name = groups[idx]->FuncName();
      auto fn_ptr = compiler_->Lookup(fn_name);
      cinn_kernel_info.fn_ptr = fn_ptr;
      // cinn_kernel_info.compiler =
      // reinterpret_cast<void*>(compiler_.release());
      cinn_kernel_info.int_args_map = groups[idx]->int_args_map;

      cinn_kernel_info_vecs.push_back(cinn_kernel_info);
    }
  }
  return cinn_kernel_info_vecs;
}

std::unique_ptr<Program> PirCompiler::Build(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<std::unique_ptr<Instruction>> instructions(groups.size());
  if (FLAGS_cinn_bucket_compile) {
    for (int i = 0; i < groups.size(); ++i) {
      group_compilation_contexts_.emplace_back(target_, groups[i], scope_);
    }
    auto worker_fn = [&](int index) {
      CompilationTask task(&group_compilation_contexts_[index]);
      task();
      instructions[index] = task.BuildInstruction();
    };
    utils::parallel_run(
        worker_fn, utils::SequenceDispatcher(0, groups.size()), -1);
  } else {
    auto op_lowerer = CreateOpLowerer<pir::GroupPtr>(target_);

    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
    for (int i = 0; i < groups.size(); ++i) {
      lowered_funcs.emplace_back(op_lowerer.Lower(groups[i]));
    }

    for (auto&& lowered_func : lowered_funcs) {
      ProcessFunction(lowered_func);
    }

    compiler_ = backends::Compiler::Create(target_);
    auto build_module = m_builder_.Build();
    compiler_->Build(build_module, "");

    instructions = BuildInstructions(groups);
  }

  // TODO(Aurelius84): Instantiate all tensors on compile-time, which is
  // controlled by 'options.with_instantiate_variables' in GraphCompiler.
  // Moreover, it's better to implement InsertBufferHandlers() logic
  // to automatically insert Malloc and Free instructions.
  for (auto& name : scope_->var_names()) {
    std::string var_name({name.data(), name.size()});
    VLOG(4) << "Instantiate " << var_name << " on compile-time";
    auto* var = scope_->Var<Tensor>(var_name);
    auto& tensor = absl::get<Tensor>(*var);
    tensor->mutable_data(target_, tensor->type());
  }
  return std::make_unique<Program>(scope_, std::move(instructions));
}

void PirCompiler::ProcessFunction(
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  for (auto&& func : lowered_funcs) {
    // for (auto&& arg : func->args) {
    //   std::string arg_name = arg.name();
    //   if (arg_name[0] == '_') arg_name = arg_name.substr(1);
    //
    //   auto* var = scope_->FindVar(arg_name);
    //   // For argument buffer not in scope, create it.
    //   if (!var && arg.is_buffer()) {
    //     auto* new_var = scope_->Var<Tensor>(arg_name);
    //     auto& tensor = absl::get<Tensor>(*new_var);
    //     std::vector<Shape::dim_t> shape;
    //     for (auto& shape_dim : arg.buffer_arg()->shape) {
    //       CHECK(shape_dim.is_constant());
    //       shape.push_back(static_cast<int>(shape_dim.get_constant()));
    //     }
    //     tensor->Resize(Shape{shape});
    //     tensor->set_type(arg.buffer_arg()->dtype);
    //   }
    // }
    m_builder_.AddFunction(func);
  }
}

std::vector<std::unique_ptr<Instruction>> PirCompiler::BuildInstructions(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<std::unique_ptr<Instruction>> instructions;
  for (int idx = 0; idx < groups.size(); ++idx) {
    auto fn_name = groups[idx]->FuncName();
    auto instr =
        std::unique_ptr<Instruction>(new Instruction(target_,
                                                     scope_.get(),
                                                     groups[idx]->input_names,
                                                     groups[idx]->output_names,
                                                     fn_name));
    VLOG(4) << "Lookup kernel name: " << fn_name;
    auto* fn_ptr = compiler_->Lookup(fn_name);
    CHECK(fn_ptr);
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), fn_name);
    // As some instruction like reduce, will generate more than one kernel.
    // So try to find the rest kernel, if it exists.
    // SetSubKernels(instr.get(), fn_name);
    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
  return instructions;
}

std::shared_ptr<Scope> BuildScope(const Target& target,
                                  const ::pir::Program& program) {
  std::unordered_set<::pir::Value> visited;
  auto scope = std::make_shared<Scope>();

  auto create_var = [&](::pir::Value value) {
    if (!(value) || !(value.type())) {
      return;
    }
    if (visited.count(value) > 0) return;
    visited.emplace(value);

    std::string name = pir::CompatibleInfo::ValueName(value);
    auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto* var = scope->Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);

    std::vector<Shape::dim_t> shape;
    for (auto i = 0; i < type_info.dims().size(); ++i) {
      shape.push_back(Shape::dim_t(type_info.dims()[i]));
    }
    tensor->Resize(Shape{shape});
    tensor->set_type(pir::CompatibleInfo::ConvertIRType(type_info.dtype()));
  };

  for (auto& op : *program.block()) {
    for (auto oprand : op.operands()) {
      create_var(oprand.source());
    }

    for (auto result : op.results()) {
      create_var(result);
    }
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
