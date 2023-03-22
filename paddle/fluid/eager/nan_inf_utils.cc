// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/nan_inf_utils.h"

#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"

#include "paddle/phi/core/compat/convert_utils.h"
DECLARE_int32(check_nan_inf_level);
namespace egr {

static std::once_flag dump_list_init_flag;

static std::unordered_set<std::string>& nan_inf_check_op_list() {
  static std::unordered_set<std::string> _check_op_list = {};
  return _check_op_list;
}

static std::unordered_set<std::string>& nan_inf_skip_op_list() {
  static std::unordered_set<std::string> _skip_op_list = {};
  return _skip_op_list;
}

static void InitDumpListFormEnv() {
  nan_inf_check_op_list();
  nan_inf_skip_op_list();
  const char* check_op_list = std::getenv("FLAGS_check_nan_inf_op_list");
  const char* skip_op_list = std::getenv("FLAGS_skip_nan_inf_op_list");

  if (check_op_list) {
    std::stringstream ss(check_op_list);
    std::string op_type;
    LOG(INFO) << "Please set op list according to the "
                 "paddle.amp.low_precision_op_list()";
    while (std::getline(ss, op_type, ',')) {
      nan_inf_check_op_list().emplace(op_type);
    }
  }

  if (skip_op_list) {
    std::stringstream ss(skip_op_list);
    std::string op_type;
    LOG(INFO) << "Please set op list according to the "
                 "paddle.amp.low_precision_op_list()";
    while (std::getline(ss, op_type, ',')) {
      nan_inf_skip_op_list().emplace(op_type);
    }
  }

  for (auto const& key : nan_inf_check_op_list()) {
    if (nan_inf_skip_op_list().count(key) != 0) {
      LOG(INFO) << "Check nan inf op list: " << key;
    }
  }
}
bool CheckOp(const std::string& api_name) {
  if (nan_inf_skip_op_list().count("all") ||
      nan_inf_skip_op_list().count(api_name)) {
    return false;
  }

  if (nan_inf_check_op_list().count("all") ||
      nan_inf_check_op_list().count(api_name)) {
    return true;
  }

  return true;
}

void CheckTensorHasNanOrInf(const std::string& api_name, const Tensor& tensor) {
  std::call_once(dump_list_init_flag, InitDumpListFormEnv);
  auto op_name = phi::TransToFluidOpName(api_name);
  if (tensor.initialized() && CheckOp(op_name)) {
    auto& tensor_name = tensor.name();
    const phi::DenseTensor* dense_tensor{nullptr};
    if (tensor.is_dense_tensor()) {
      dense_tensor = static_cast<const phi::DenseTensor*>(tensor.impl().get());
    } else if (tensor.is_selected_rows()) {
      dense_tensor = &(
          static_cast<const phi::SelectedRows*>(tensor.impl().get())->value());
    } else {
      VLOG(10) << "Only DenseTensor or SelectedRows need to check, "
               << tensor_name << " is no need.";
      return;
    }

    // dump data
    if (FLAGS_check_nan_inf_level == 4) {
      auto dir_path = paddle::framework::details::GetNanPath();
      std::string adr_name = paddle::string::Sprintf("%d", tensor.impl());
      std::string folder_path = dir_path + "tensor_dump/";
      std::string mkdir_cmd = "mkdir -p " + folder_path;
      PADDLE_ENFORCE_EQ(system(mkdir_cmd.c_str()),
                        0,
                        paddle::platform::errors::NotFound(
                            "Cannot create folder %s", folder_path));

      std::string str_file_name = folder_path + api_name + "_" + adr_name;
      std::ofstream fout(str_file_name, std::ios::out | std::ios::binary);

      PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                        true,
                        paddle::platform::errors::Unavailable(
                            "Cannot open %s to save tensor.", str_file_name));

      VLOG(4) << "The Dump file path is " << str_file_name;
      paddle::framework::SerializeToStream(fout, *dense_tensor);
      fout.close();
      return;
    }

    auto& place = dense_tensor->place();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      paddle::framework::details::tensor_check<phi::GPUContext>(
          api_name, tensor_name, *dense_tensor, place);
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "Tensor[%s] use gpu place. PaddlePaddle must compile with GPU.",
          tensor_name));
#endif
      return;
    }
    paddle::framework::details::tensor_check<phi::CPUContext>(
        api_name, tensor_name, *dense_tensor, place);
  }
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfTwoTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfThreeTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfFourTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<3>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfFiveTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<3>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<4>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfSixTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<3>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<4>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<5>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const std::vector<Tensor>& tensors) {
  for (auto& tensor : tensors) {
    CheckTensorHasNanOrInf(api_name, tensor);
  }
}

void CheckTensorHasNanOrInf(
    const std::string& api_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>& tensors) {
  for (auto& tensor_vector : tensors) {
    CheckTensorHasNanOrInf(api_name, tensor_vector);
  }
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfTensorAndVector& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
}

}  // namespace egr
