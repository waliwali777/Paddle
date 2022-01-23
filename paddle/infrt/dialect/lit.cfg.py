# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lit.formats
import os

config.name = "MLIR tests"
config.test_format = lit.formats.ShTest(True)
build_dir = "/home/chunwei/project/Paddle/cmake-build-debug"
config.llvm_tools_dir = os.path.join(build_dir, "third_party/install/llvm/bin")
config.llvm_tools_dir = os.path.join(build_dir, "/third_party/install/llvm/lib")
test_bin = os.path.join(build_dir, "paddle/infrt/dialect/")
llvm_bin = os.path.join(build_dir, "third_party/install/llvm/bin/")
config.environment['PATH'] = os.path.pathsep.join(
    (test_bin, llvm_bin, config.environment['PATH']))

config.suffixes = ['.mlir']
