// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/codegen_cuda_util.h"

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"

PD_DECLARE_bool(cinn_bucket_compile);
namespace cinn {
namespace backends {

std::tuple<ir::Module, ir::Module> SplitDeviceAndHostModule(ir::Module module,
                                                            Target target) {
  if (FLAGS_cinn_bucket_compile) {
    detail::CollectBucketStrategyHostFunctionVisitor visitor(module->name,
                                                             target);
    Expr expr(module);
    return visitor(&expr);
  }
  detail::CollectHostFunctionVisitor visitor(module->name, target);
  Expr expr(module);
  return visitor(&expr);
}

struct PredicatePrinter : public ir::IrPrinter {
  explicit PredicatePrinter(std::ostream &os) : ir::IrPrinter(os) {}

 private:
  void Visit(const ir::Add *x) { PrintBinaryOp("ADD", x); }
  void Visit(const ir::Sub *x) { PrintBinaryOp("SUB", x); }
  void Visit(const ir::Mul *x) { PrintBinaryOp("MUL", x); }
  void Visit(const ir::Div *x) { PrintBinaryOp("DIV", x); }
  void Visit(const ir::Mod *x) { PrintBinaryOp("MOD", x); }
  void Visit(const ir::EQ *x) { PrintBinaryOp("EQ", x); }
  void Visit(const ir::NE *x) { PrintBinaryOp("NE", x); }
  void Visit(const ir::LT *x) { PrintBinaryOp("LT", x); }
  void Visit(const ir::LE *x) { PrintBinaryOp("LE", x); }
  void Visit(const ir::GT *x) { PrintBinaryOp("GT", x); }
  void Visit(const ir::GE *x) { PrintBinaryOp("GE", x); }
  void Visit(const ir::And *x) { PrintBinaryOp("AND", x); }
  void Visit(const ir::Or *x) { PrintBinaryOp("OR", x); }

  template <typename IRN>
  void PrintBinaryOp(const std::string &op, const ir::BinaryOpNode<IRN> *x) {
    str_ += "_FPA_";
    ir::IrPrinter::Visit(x->a());
    str_ += op;
    ir::IrPrinter::Visit(x->b());
    str_ += "_BPA_";
  }
};

std::string Predicate2String(ir::Expr predicate) {
  std::stringstream ss;
  PredicatePrinter cond_printer(ss);
  cond_printer.Print(predicate);
  return ss.str();
}

std::string
detail::CollectBucketStrategyHostFunctionVisitor::GenDeviceKernelName(
    const std::string &fn_name, ir::Expr predicate) {
  std::string cond_str = Predicate2String(predicate);
  VLOG(3) << "predicate string: " << cond_str;
  return fn_name + "__COND_" + cond_str + "__kernel";
}

void detail::CollectBucketStrategyHostFunctionVisitor::ProcessLoweredFunc(
    ir::Expr func, ir::Expr predicate) {
  VLOG(4) << "Process Lowered Func" << func;
  ir::_LoweredFunc_ *func_node = func.as_lowered_func();
  CHECK(func_node);
  if (!func_node->cuda_axis_info.valid()) {
    func_node->cuda_axis_info.set_valid(true);
  }
  // process device func
  device_module_builder.AddFunctionWithoutOptim(
      CreateDeviceFunction(func, predicate).as_lowered_func_ref());
  // process host func
  ir::Var kernel_ptr(GenDeviceKernelName(func_node->name, predicate),
                     type_of<std::string>());

  // shared_mem_bytes Can be calculated after codegen_cuda_dev buffer creation
  // however, this make CodeGenCUDA_Dev before spliting the host and device
  // module Maybe we could reorder the process.
  CodeGenCUDA_Dev codegen_dev(cinn::common::DefaultNVGPUTarget());
  codegen_dev.Compile(ir::LoweredFunc(func.as_lowered_func_ref()));
  Expr shared_mem_bytes = codegen_dev.GetDynSharedMemOffset();

  VLOG(6) << "Add a call node for func_node->name " << func_node->name << "\n"
          << "grid_dim: (" << func_node->cuda_axis_info.grid_dim(0) << ", "
          << func_node->cuda_axis_info.grid_dim(1) << ", "
          << func_node->cuda_axis_info.grid_dim(2) << "), "
          << "block_dim: (" << func_node->cuda_axis_info.block_dim(0) << ", "
          << func_node->cuda_axis_info.block_dim(1) << ", "
          << func_node->cuda_axis_info.block_dim(2) << "), "
          << "shared_mem: " << shared_mem_bytes;

  const char *call_kernel;
  if (target_ == common::DefaultNVGPUTarget()) {
    call_kernel = runtime::intrinsic::call_cuda_kernel;
  } else if (target_.language == common::Target::Language::sycl) {
    call_kernel = runtime::intrinsic::call_sycl_kernel;
  } else if (target_.language == common::Target::Language::hip) {
    call_kernel = runtime::intrinsic::call_hip_kernel;
  }
  ir::Expr call_extern_api =
      ir::Call::Make(Void(),
                     call_kernel,
                     {kernel_ptr,
                      kernel_args_,
                      kernel_args_num_,
                      func_node->cuda_axis_info.grid_dim(0),   // grid_x
                      func_node->cuda_axis_info.grid_dim(1),   // grid_y
                      func_node->cuda_axis_info.grid_dim(2),   // grid_z
                      func_node->cuda_axis_info.block_dim(0),  // block_x
                      func_node->cuda_axis_info.block_dim(1),  // block_y
                      func_node->cuda_axis_info.block_dim(2),  // block_z
                      shared_mem_bytes,                        // shared_mem
                      kernel_stream_},
                     {},
                     ir::CallType::Extern,
                     ir::FunctionRef(),
                     0);
  buckets_.emplace_back(ir::IfThenElse::Make(predicate, call_extern_api));
}

void detail::CollectBucketStrategyHostFunctionVisitor::ProcessArgs(
    ir::Expr func) {
  std::vector<ir::Argument> args = func.as_lowered_func_ref()->args;
  for (int i = 0; i < args.size(); ++i) {
    if (args[i].is_var()) {
      ir::Expr call_get_value_in_kernel_args =
          ir::Call::Make(Int(64),
                         runtime::intrinsic::get_value_in_cuda_kernel_args,
                         {kernel_args_, ir::Expr(i)},
                         {},
                         ir::CallType::Extern,
                         ir::FunctionRef(),
                         0);
      ir::Expr let_symbol = ir::Expr(args[i].var_arg());
      let_symbol->set_type(type_of<int64_t>());
      ir::Expr stmt = ir::Let::Make(let_symbol, call_get_value_in_kernel_args);
      arg_defs_.push_back(stmt);
    }
  }
}

Expr detail::CollectBucketStrategyHostFunctionVisitor::CreateDeviceFunction(
    ir::Expr expr, ir::Expr predicate) {
  auto copied = ir::ir_utils::IRCopy(expr);
  auto *lowered_func = copied.as_lowered_func();
  lowered_func->name = GenDeviceKernelName(lowered_func->name, predicate);
  return copied;
}

}  // namespace backends
}  // namespace cinn
