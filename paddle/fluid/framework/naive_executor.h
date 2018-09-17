#pragma once

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

/*
 * Simple, intuitive and effective. Only single thread is supported, and
 * currently designed for inference.
 */
class NaiveExecutor {
 public:
  explicit NaiveExecutor(const platform::Place& place) : place_(place) {}

  // Create child scope.
  // Create variables.
  void Prepare(Scope *parent_scope, const ProgramDesc &program_desc, int block_id);

  // Run all the operators.
  void Run();

  // Get an tensor to operating directly, without the need for feed_ops.
  LoDTensor* FindTensor(const std::string& name);

  Scope *scope() { return scope_; }

 protected:
  void CreateVariables(const ProgramDesc& desc, Scope* scope, int block_id);

  void CreateOps(const ProgramDesc &desc, int block_id);

 private:
  const platform::Place place_;
  // Catch the required resource to avoid recreate.
  std::vector<std::unique_ptr<OperatorBase>> ops_;
  Scope *scope_;
};

}  // namespace framework
}  // namespace paddle
