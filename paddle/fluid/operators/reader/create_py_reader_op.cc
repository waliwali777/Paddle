// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/ddim.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class CreatePyReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {}
};

class CreatePyReaderOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddInput("blocking_queue",
             "Name of the `LoDTensorBlockingQueueHolder` variable");

    AddAttr<int>("device_index", "The device index this reader offers data")
        .SetDefault(0);

    AddAttr<int>("device_count",
                 "The total device number this reader offers data")
        .SetDefault(1);

    AddComment(R"DOC(
      Create PyReader to support phi::DenseTensor data feeding in Python side.
      )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_py_reader,
                              reader::CreatePyReaderOp,
                              reader::CreatePyReaderOpMaker);
