// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <vector>
#include "lite/model_parser/base/io.h"
#include "lite/model_parser/base/param_desc.h"
#include "lite/model_parser/pb/utils.h"

namespace paddle {
namespace lite {
namespace pb {

class TensorInfoReader : public TensorInfoReadAPI {
 public:
  TensorInfoReader(model_parser::BytesReader* reader,
                   model_parser::Buffer* buffer) {
    CHECK(reader);
    CHECK(buffer);
    int32_t size = reader->ReadForward<int32_t>();
    buffer->ReallocateDownward(size);
    reader->ReadForward(buffer->data(), size);
    CHECK(desc_.ParseFromArray(buffer->data(), size))
        << "Cannot parse tensor desc";
  }
  std::vector<int64_t> Dim() const override {
    std::vector<int64_t> dims_vec;
    std::copy(
        desc_.dims().begin(), desc_.dims().end(), std::back_inserter(dims_vec));
    return dims_vec;
  }
  VarDataType GetDataType() const override {
    return ConvertVarType(desc_.data_type());
  }

 private:
  framework::proto::VarType::TensorDesc desc_;
};

class TensorInfoWriter : public TensorInfoWriteAPI {
 public:
  TensorInfoWriter(model_parser::BytesWriter* writer,
                   model_parser::Buffer* buffer)
      : writer_(writer), buffer_(buffer) {
    CHECK(writer_);
    CHECK(buffer_);
  }
  void SetDim(const std::vector<int64_t>& dim) override { dim_ = dim; }
  void SetDataType(VarDataType data_type) override { data_type_ = data_type; }
  void Sync() override {
    desc_.set_data_type(ConvertVarType(data_type_));
    auto* pb_dims = desc_.mutable_dims();
    pb_dims->Resize(dim_.size(), 0);
    std::copy(dim_.begin(), dim_.end(), pb_dims->begin());
    int32_t desc_size = desc_.ByteSizeLong();
    writer_->WriteForward<int32_t>(desc_size);
    buffer_->ReallocateDownward(desc_.ByteSizeLong());
    desc_.SerializeToArray(buffer_->data(), buffer_->size());
    writer_->WriteForward(buffer_->data(), buffer_->size());
  }

 private:
  framework::proto::VarType::TensorDesc desc_;
  std::vector<int64_t> dim_;
  VarDataType data_type_;
  model_parser::BytesWriter* writer_;
  model_parser::Buffer* buffer_;
};

}  // namespace pb
}  // namespace lite
}  // namespace paddle
