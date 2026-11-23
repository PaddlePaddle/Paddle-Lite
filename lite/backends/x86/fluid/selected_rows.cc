/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/fluid/selected_rows.h"

namespace paddle {
namespace lite {
namespace fluid {

#ifndef LITE_ON_TINY_PUBLISH
void SerializeToStream(std::ostream& os,
                       const SelectedRows& selected_rows,
                       const lite::Context<lite::TargetType::kX86>& dev_ctx) {
  {
    // the 1st field, unit32_t version for SelectedRows
    uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {
    // the 2st field, rows information
    auto& rows = selected_rows.rows();
    uint64_t size = static_cast<uint64_t>(rows.size());
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (uint64_t i = 0; i < size; ++i) {
      os.write(reinterpret_cast<const char*>(&rows[i]), sizeof(rows[i]));
    }
  }
  {
    // the 3st field, the height of SelectedRows
    int64_t height = selected_rows.height();
    os.write(reinterpret_cast<const char*>(&height), sizeof(height));
  }
  // the 4st field, Tensor data
  TensorToStream(os, selected_rows.value());
}

void DeserializeFromStream(
    std::istream& is,
    SelectedRows* selected_rows,
    const lite::Context<lite::TargetType::kX86>& dev_ctx) {
  {
    // the 1st field, unit32_t version for SelectedRows
    uint32_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    CHECK_EQ(version, 0U) << "Only version 0 is supported";
  }
  {
    // the 2st field, rows information
    uint64_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    auto& rows = *selected_rows->mutable_rows();
    rows.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
      is.read(reinterpret_cast<char*>(&rows[i]), sizeof(int64_t));
    }
  }
  {
    // the 3st field, the height of the SelectedRows
    int64_t height;
    is.read(reinterpret_cast<char*>(&height), sizeof(int64_t));
    selected_rows->set_height(height);
  }
  // the 4st field, tensor which contains the data
  TensorFromStream(is, selected_rows->mutable_value());
}
#endif

bool SelectedRows::HasKey(int64_t key) const {
  return std::find(rows_.begin(), rows_.end(), key) == rows_.end() ? false
                                                                   : true;
}

int64_t SelectedRows::AutoGrownIndex(int64_t key,
                                     bool auto_grown,
                                     bool is_test) {
  if (is_test) {
    auto iter = id_to_index_.find(key);
    if (iter == id_to_index_.end()) {
      return -1;
    } else {
      return iter->second;
    }
  }

  auto iter = id_to_index_.find(key);
  if (iter == id_to_index_.end()) {
    if (!auto_grown) {
      LOG(FATAL) << "id " << key << " not in table";
    }
    auto index = static_cast<int64_t>(rows_.size());
    rows_.push_back(key);
    id_to_index_[key] = index;
    return index;
  } else {
    return iter->second;
  }
}

void SelectedRows::SyncIndex() {
  id_to_index_.clear();
  for (size_t i = 0; i < rows_.size(); ++i) {
    id_to_index_[rows_[i]] = static_cast<int64_t>(i);
  }
}

void SelectedRows::Get(const lite::Tensor& ids,
                       lite::Tensor* value,
                       bool auto_grown,
                       bool is_test) {
  auto* ids_data = ids.data<int64_t>();
  int64_t ids_size = ids.numel();

  auto value_dims = value_->dims();
  auto output_dims = value->dims();
  if (output_dims.size() == 0) {
    output_dims = value_dims;
    output_dims[0] = ids_size;
    value->Resize(output_dims);
  } else {
    CHECK_EQ(output_dims.size(), value_dims.size());
    CHECK_EQ(output_dims[0], ids_size);
    for (size_t i = 1; i < value_dims.size(); ++i) {
      CHECK_EQ(output_dims[i], value_dims[i]);
    }
  }

  int64_t row_width = value_->numel() / value_dims[0];
  auto* src_data = value_->data<float>();
  auto* dst_data = value->mutable_data<float>();

  for (int64_t i = 0; i < ids_size; ++i) {
    int64_t row_index = AutoGrownIndex(ids_data[i], auto_grown, is_test);
    if (row_index == -1) {
      memset(dst_data + i * row_width, 0, sizeof(float) * row_width);
    } else {
      memcpy(dst_data + i * row_width,
             src_data + row_index * row_width,
             sizeof(float) * row_width);
    }
  }
}

}  // namespace fluid
}  // namespace lite
}  // namespace paddle
