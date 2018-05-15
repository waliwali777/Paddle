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

#pragma once

#include <string>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/framework.pb.h"
namespace paddle {
namespace framework {

// this class not only make proto but also init attribute checkers.
class OpProtoAndCheckerMaker {
 public:
  virtual void Make() = 0;

  virtual ~OpProtoAndCheckerMaker() {
    CHECK(validated_) << "should call Validate after build";
  }

  void SetProto(proto::OpProto *proto) { proto_ = proto; }

  void SetChecker(OpAttrChecker *attr_checker) { op_checker_ = attr_checker; }

  void Validate();

 protected:
  struct VariableBuilder {
    proto::OpProto::Var *var_;

    VariableBuilder &AsDuplicable() {
      var_->set_duplicable(true);
      return *this;
    }

    VariableBuilder &AsIntermediate() {
      var_->set_intermediate(true);
      return *this;
    }

    VariableBuilder &AsDispensable() {
      var_->set_dispensable(true);
      return *this;
    }

    VariableBuilder &Reuse(const std::string &name) {
      var_->set_reuse(name);
      return *this;
    }
  };

  VariableBuilder AddInput(const std::string &name, const std::string &comment);

  VariableBuilder AddOutput(const std::string &name,
                            const std::string &comment);

  template <typename T>
  TypedAttrChecker<T> &AddAttr(const std::string &name,
                               const std::string &comment,
                               bool generated = false) {
    auto *attr = proto_->add_attrs();
    attr->set_name(name);
    attr->set_comment(comment);
    attr->set_generated(generated);
    attr->set_type(AttrTypeID<T>());
    return op_checker_->AddAttrChecker<T>(name);
  }

  void AddComment(const std::string &comment) { proto_->set_comment(comment); }

 private:
  void CheckNoDuplicatedInOutAttrs();

  void CheckReuseVars();

  proto::OpProto *proto_;
  OpAttrChecker *op_checker_;
  bool validated_{false};
};
}  // namespace framework
}  // namespace paddle
