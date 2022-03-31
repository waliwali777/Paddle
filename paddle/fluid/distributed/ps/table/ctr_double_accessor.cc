// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/ctr_double_accessor.h"
#include <gflags/gflags.h>
#include "glog/logging.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

int DownpourCtrDoubleAccessor::Initialize() {
  auto name = _config.embed_sgd_param().name();
  _embed_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

  name = _config.embedx_sgd_param().name();
  _embedx_sgd_rule = CREATE_PSCORE_CLASS(SparseValueSGDRule, name);
  _embedx_sgd_rule->load_config(_config.embedx_sgd_param(),
                                _config.embedx_dim());

  _show_click_decay_rate = _config.ctr_accessor_param().show_click_decay_rate();
  _ssd_unseenday_threshold =
      _config.ctr_accessor_param().ssd_unseenday_threshold();

  return 0;
}

void DownpourCtrDoubleAccessor::SetTableInfo(AccessorInfo& info) {
  info.dim = Dim();
  info.size = Size();
  info.select_dim = SelectDim();
  info.select_size = SelectSize();
  info.update_dim = UpdateDim();
  info.update_size = UpdateSize();
  info.mf_size = MFSize();
}

size_t DownpourCtrDoubleAccessor::GetTableInfo(InfoKey key) {
  switch (key) {
    case DIM:
      return Dim();
    case SIZE:
      return Size();
    case SELECT_DIM:
      return SelectDim();
    case SELECT_SIZE:
      return SelectSize();
    case UPDATE_DIM:
      return UpdateDim();
    case UPDATE_SIZE:
      return UpdateSize();
    case MF_SIZE:
      return MFSize();
    default:
      return 0;
  }
  return 0;
}

size_t DownpourCtrDoubleAccessor::Dim() {
  auto embedx_dim = _config.embedx_dim();
  return DownpourCtrDoubleFeatureValue::Dim(embedx_dim);
}
size_t DownpourCtrDoubleAccessor::DimSize(size_t dim) {
  auto embedx_dim = _config.embedx_dim();
  return DownpourCtrDoubleFeatureValue::DimSize(dim, embedx_dim);
}
size_t DownpourCtrDoubleAccessor::Size() {
  auto embedx_dim = _config.embedx_dim();
  return DownpourCtrDoubleFeatureValue::Size(embedx_dim);
}
size_t DownpourCtrDoubleAccessor::MFSize() {
  return (_config.embedx_dim() + 1) * sizeof(float);  // embedx embedx_g2sum
}
// pull value
size_t DownpourCtrDoubleAccessor::SelectDim() {
  auto embedx_dim = _config.embedx_dim();
  return 3 + embedx_dim;
}
size_t DownpourCtrDoubleAccessor::SelectDimSize(size_t dim) {
  return sizeof(float);
}
size_t DownpourCtrDoubleAccessor::SelectSize() {
  return SelectDim() * sizeof(float);
}
// push value
size_t DownpourCtrDoubleAccessor::UpdateDim() {
  auto embedx_dim = _config.embedx_dim();
  return 4 + embedx_dim;
}
size_t DownpourCtrDoubleAccessor::UpdateDimSize(size_t dim) {
  return sizeof(float);
}
size_t DownpourCtrDoubleAccessor::UpdateSize() {
  return UpdateDim() * sizeof(float);
}
bool DownpourCtrDoubleAccessor::Shrink(float* value) {
  // auto base_threshold = _config.ctr_accessor_param().base_threshold();
  // auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  // auto delete_threshold = _config.ctr_accessor_param().delete_threshold();
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delete_after_unseen_days =
      _config.ctr_accessor_param().delete_after_unseen_days();
  auto delete_threshold = _config.ctr_accessor_param().delete_threshold();
  // time_decay first
  DownpourCtrDoubleFeatureValue::Show(value) *= _show_click_decay_rate;
  DownpourCtrDoubleFeatureValue::Click(value) *= _show_click_decay_rate;
  // shrink after
  auto score = show_click_score(DownpourCtrDoubleFeatureValue::Show(value),
                                DownpourCtrDoubleFeatureValue::Click(value));
  auto unseen_days = DownpourCtrDoubleFeatureValue::unseen_days(value);
  if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
    return true;
  }
  return false;
}
bool DownpourCtrDoubleAccessor::save_ssd(float* value) {
  if (DownpourCtrDoubleFeatureValue::unseen_days(value) >
      _ssd_unseenday_threshold) {
    return true;
  }
  return false;
}
// bool DownpourCtrDoubleAccessor::save_cache(
//         float* value, int param, double global_cache_threshold) {
//     auto base_threshold = _config.ctr_accessor_param().base_threshold();
//     auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
//     if (show_click_score(DownpourCtrDoubleFeatureValue::Show(value),
//     DownpourCtrDoubleFeatureValue::Click(value)) >= base_threshold
//         && DownpourCtrDoubleFeatureValue::unseen_days(value) <=
//         delta_keep_days) {
//         return DownpourCtrDoubleFeatureValue::Show(value) >
//         global_cache_threshold;
//     }
//     return false;
// }
bool DownpourCtrDoubleAccessor::Save(float* value, int param) {
  // auto base_threshold = _config.ctr_accessor_param().base_threshold();
  // auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  // auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    // save all
    case 0: {
      return true;
    }
    // save xbox delta
    case 1:
    // save xbox base
    case 2: {
      if (show_click_score(DownpourCtrDoubleFeatureValue::Show(value),
                           DownpourCtrDoubleFeatureValue::Click(value)) >=
              base_threshold &&
          DownpourCtrDoubleFeatureValue::delta_score(value) >=
              delta_threshold &&
          DownpourCtrDoubleFeatureValue::unseen_days(value) <=
              delta_keep_days) {
        // do this after save, because it must not be modified when retry
        if (param == 2) {
          DownpourCtrDoubleFeatureValue::delta_score(value) = 0;
        }
        return true;
      } else {
        return false;
      }
    }
    // already decayed in shrink
    case 3: {
      // DownpourCtrFeatureValue::Show(value) *= _show_click_decay_rate;
      // DownpourCtrFeatureValue::Click(value) *= _show_click_decay_rate;
      // do this after save, because it must not be modified when retry
      // DownpourCtrDoubleFeatureValue::unseen_days(value)++;
      return true;
    }
    default:
      return true;
  };
}

void DownpourCtrDoubleAccessor::UpdateStatAfterSave(float* value,
                                                       int param) {
  auto base_threshold = _config.ctr_accessor_param().base_threshold();
  auto delta_threshold = _config.ctr_accessor_param().delta_threshold();
  auto delta_keep_days = _config.ctr_accessor_param().delta_keep_days();
  if (param == 2) {
    delta_threshold = 0;
  }
  switch (param) {
    case 1: {
      if (show_click_score(DownpourCtrDoubleFeatureValue::Show(value),
                           DownpourCtrDoubleFeatureValue::Click(value)) >=
              base_threshold &&
          DownpourCtrDoubleFeatureValue::delta_score(value) >=
              delta_threshold &&
          DownpourCtrDoubleFeatureValue::unseen_days(value) <=
              delta_keep_days) {
        DownpourCtrDoubleFeatureValue::delta_score(value) = 0;
      }
    }
      return;
    case 3: {
      DownpourCtrDoubleFeatureValue::unseen_days(value)++;
    }
      return;
    default:
      return;
  };
}

int32_t DownpourCtrDoubleAccessor::Create(float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* value = values[value_item];
    value[DownpourCtrDoubleFeatureValue::unseen_days_index()] = 0;
    value[DownpourCtrDoubleFeatureValue::delta_score_index()] = 0;
    *(double*)(value + DownpourCtrDoubleFeatureValue::ShowIndex()) = 0;
    *(double*)(value + DownpourCtrDoubleFeatureValue::ClickIndex()) = 0;
    value[DownpourCtrDoubleFeatureValue::SlotIndex()] = -1;
    _embed_sgd_rule->init_value(
        value + DownpourCtrDoubleFeatureValue::Embed_W_Index(),
        value + DownpourCtrDoubleFeatureValue::embed_g2sum_index());
    _embedx_sgd_rule->init_value(
        value + DownpourCtrDoubleFeatureValue::Embedx_W_Index(),
        value + DownpourCtrDoubleFeatureValue::embedx_g2sum_index(), false);
  }
  return 0;
}
bool DownpourCtrDoubleAccessor::NeedExtendMF(float* value) {
  auto show =
      ((double*)(value + DownpourCtrDoubleFeatureValue::ShowIndex()))[0];
  auto click =
      ((double*)(value + DownpourCtrDoubleFeatureValue::ClickIndex()))[0];
  // float score = (show - click) * _config.ctr_accessor_param().nonclk_coeff()
  auto score = (show - click) * _config.ctr_accessor_param().nonclk_coeff() +
               click * _config.ctr_accessor_param().click_coeff();
  //+ click * _config.ctr_accessor_param().click_coeff();
  return score >= _config.embedx_threshold();
}
// from DownpourCtrFeatureValue to DownpourCtrPullValue
int32_t DownpourCtrDoubleAccessor::Select(float** select_values,
                                          const float** values, size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* select_value = select_values[value_item];
    float* value = const_cast<float*>(values[value_item]);
    select_value[DownpourCtrDoublePullValue::ShowIndex()] =
        (float)*(double*)(value + DownpourCtrDoubleFeatureValue::ShowIndex());
    select_value[DownpourCtrDoublePullValue::ClickIndex()] =
        (float)*(double*)(value + DownpourCtrDoubleFeatureValue::ClickIndex());
    select_value[DownpourCtrDoublePullValue::Embed_W_Index()] =
        value[DownpourCtrDoubleFeatureValue::Embed_W_Index()];
    memcpy(select_value + DownpourCtrDoublePullValue::Embedx_W_Index(),
           value + DownpourCtrDoubleFeatureValue::Embedx_W_Index(),
           embedx_dim * sizeof(float));
  }
  return 0;
}
// from DownpourCtrPushValue to DownpourCtrPushValue
// first dim: item
// second dim: field num
int32_t DownpourCtrDoubleAccessor::Merge(float** update_values,
                                         const float** other_update_values,
                                         size_t num) {
  auto embedx_dim = _config.embedx_dim();
  size_t total_dim = DownpourCtrDoublePushValue::Dim(embedx_dim);
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* other_update_value = other_update_values[value_item];
    /**(double*)(update_value + DownpourCtrDoublePushValue::ShowIndex()) +=
    *(double*)(other_update_value + DownpourCtrDoublePushValue::ShowIndex());
    *(double*)(update_value + DownpourCtrDoublePushValue::ClickIndex()) +=
    *(double*)(other_update_value + DownpourCtrDoublePushValue::ClickIndex());
    for (auto i = 3u; i < total_dim; ++i) {
        update_value[i] += other_update_value[i];
    }*/
    for (auto i = 0u; i < total_dim; ++i) {
      if (i != DownpourCtrDoublePushValue::SlotIndex()) {
        update_value[i] += other_update_value[i];
      }
    }
  }
  return 0;
}
// from DownpourCtrPushValue to DownpourCtrFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourCtrDoubleAccessor::Update(float** update_values,
                                          const float** push_values,
                                          size_t num) {
  auto embedx_dim = _config.embedx_dim();
  for (size_t value_item = 0; value_item < num; ++value_item) {
    float* update_value = update_values[value_item];
    const float* push_value = push_values[value_item];
    float push_show = push_value[DownpourCtrDoublePushValue::ShowIndex()];
    float push_click = push_value[DownpourCtrDoublePushValue::ClickIndex()];
    float slot = push_value[DownpourCtrDoublePushValue::SlotIndex()];
    *(double*)(update_value + DownpourCtrDoubleFeatureValue::ShowIndex()) +=
        (double)push_show;
    *(double*)(update_value + DownpourCtrDoubleFeatureValue::ClickIndex()) +=
        (double)push_click;
    update_value[DownpourCtrDoubleFeatureValue::SlotIndex()] = slot;
    update_value[DownpourCtrDoubleFeatureValue::delta_score_index()] +=
        (push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
        push_click * _config.ctr_accessor_param().click_coeff();
    //(push_show - push_click) * _config.ctr_accessor_param().nonclk_coeff() +
    // push_click * _config.ctr_accessor_param().click_coeff();
    update_value[DownpourCtrDoubleFeatureValue::unseen_days_index()] = 0;
    _embed_sgd_rule->update_value(
        update_value + DownpourCtrDoubleFeatureValue::Embed_W_Index(),
        update_value + DownpourCtrDoubleFeatureValue::embed_g2sum_index(),
        push_value + DownpourCtrDoublePushValue::Embed_G_Index(), push_show);
    _embedx_sgd_rule->update_value(
        update_value + DownpourCtrDoubleFeatureValue::Embedx_W_Index(),
        update_value + DownpourCtrDoubleFeatureValue::embedx_g2sum_index(),
        push_value + DownpourCtrDoublePushValue::Embedx_G_Index(), push_show);
  }
  return 0;
}
bool DownpourCtrDoubleAccessor::CreateValue(int stage, const float* value) {
  // stage == 0, pull
  // stage == 1, push
  if (stage == 0) {
    return true;
  } else if (stage == 1) {
    auto show = DownpourCtrDoublePushValue::Show(const_cast<float*>(value));
    auto click = DownpourCtrDoublePushValue::Click(const_cast<float*>(value));
    auto score = show_click_score(show, click);
    if (score <= 0) {
      return false;
    }
    if (score >= 1) {
      return true;
    }
    return local_uniform_real_distribution<float>()(local_random_engine()) <
           score;
  } else {
    return true;
  }
}
double DownpourCtrDoubleAccessor::show_click_score(double show, double click) {
  // auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  // auto click_coeff = _config.ctr_accessor_param().click_coeff();
  auto nonclk_coeff = _config.ctr_accessor_param().nonclk_coeff();
  auto click_coeff = _config.ctr_accessor_param().click_coeff();
  return (show - click) * nonclk_coeff + click * click_coeff;
}
std::string DownpourCtrDoubleAccessor::ParseToString(const float* v,
                                                       int param_size) {
  thread_local std::ostringstream os;
  os.clear();
  os.str("");
  os << v[0] << " " << v[1] << " " << (float)((double*)(v + 2))[0] << " "
     << (float)((double*)(v + 4))[0] << " " << v[6] << " " << v[7] << " "
     << v[8];
  auto show = DownpourCtrDoubleFeatureValue::Show(const_cast<float*>(v));
  auto click = DownpourCtrDoubleFeatureValue::Click(const_cast<float*>(v));
  auto score = show_click_score(show, click);
  if (score >= _config.embedx_threshold() && param_size > 9) {
    os << " " << v[9];
    for (auto i = 0; i < _config.embedx_dim(); ++i) {
      os << " " << v[10 + i];
    }
  }
  return os.str();
}
int DownpourCtrDoubleAccessor::ParseFromString(const std::string& str,
                                                 float* value) {
  int embedx_dim = _config.embedx_dim();
  float data_buff[Dim() + 2];
  float* data_buff_ptr = data_buff;
  _embedx_sgd_rule->init_value(
      data_buff_ptr + DownpourCtrDoubleFeatureValue::Embedx_W_Index(),
      data_buff_ptr + DownpourCtrDoubleFeatureValue::embedx_g2sum_index());
  auto str_len = paddle::string::str_to_float(str.data(), data_buff_ptr);
  CHECK(str_len >= 6) << "expect more than 6 real:" << str_len;
  int show_index = DownpourCtrDoubleFeatureValue::ShowIndex();
  int click_index = DownpourCtrDoubleFeatureValue::ClickIndex();
  int embed_w_index = DownpourCtrDoubleFeatureValue::Embed_W_Index();
  // no slot, embedx
  int value_dim = Dim();
  int embedx_g2sum_index = DownpourCtrDoubleFeatureValue::embedx_g2sum_index();
  value[DownpourCtrDoubleFeatureValue::SlotIndex()] = -1;
  // other case
  if (str_len == (value_dim - 1)) {
    // copy unseen_days..delta_score
    memcpy(value, data_buff_ptr, show_index * sizeof(float));
    // copy show & click
    *(double*)(value + show_index) = (double)data_buff_ptr[2];
    *(double*)(value + click_index) = (double)data_buff_ptr[3];
    // copy others
    value[DownpourCtrDoubleFeatureValue::Embed_W_Index()] = data_buff_ptr[4];
    value[DownpourCtrDoubleFeatureValue::embed_g2sum_index()] =
        data_buff_ptr[5];
    memcpy(value + embedx_g2sum_index, data_buff_ptr + 6,
           (embedx_dim + 1) * sizeof(float));
  } else {
    // copy unseen_days..delta_score
    memcpy(value, data_buff_ptr, show_index * sizeof(float));
    // copy show & click
    *(double*)(value + show_index) = (double)data_buff_ptr[2];
    *(double*)(value + click_index) = (double)data_buff_ptr[3];
    // copy embed_w..embedx_w
    memcpy(value + embed_w_index, data_buff_ptr + 4,
           (str_len - 4) * sizeof(float));
  }
  if (str_len == (value_dim - 1) || str_len == 6) {
    str_len += 1;
  }
  return str_len + 2;
}

}  // namespace distributed
}  // namespace paddle
