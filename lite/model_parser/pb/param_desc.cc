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

#include "lite/model_parser/pb/param_desc.h"

namespace paddle {
namespace lite {
namespace pb {

TensorInfoReader::TensorInfoReader(model_parser::ByteReader* reader,
                                   model_parser::Buffer* buffer) {
  CHECK(reader);
  CHECK(buffer);
  int32_t size = reader->ReadForward<int32_t>();
  buffer->ResetLazy(size);
  reader->ReadForward(buffer->data(), size);
  CHECK(desc_.ParseFromArray(buffer->data(), size))
      << "Cannot parse tensor desc";
}

void TensorInfoWriter::Sync() {
  desc_.set_data_type(ConvertVarType(data_type_));
  auto* pb_dims = desc_.mutable_dims();
  pb_dims->Resize(dim_.size(), 0);
  std::copy(dim_.begin(), dim_.end(), pb_dims->begin());
  int32_t desc_size = desc_.ByteSizeLong();
  writer_->WriteForward<int32_t>(desc_size);
  buffer_->ResetLazy(desc_.ByteSizeLong());
  desc_.SerializeToArray(buffer_->data(), buffer_->size());
  writer_->WriteForward(buffer_->data(), buffer_->size());
}

}  // namespace pb
}  // namespace lite
}  // namespace paddle
