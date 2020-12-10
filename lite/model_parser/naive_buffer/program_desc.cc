// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/model_parser/naive_buffer/program_desc.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

size_t ProgramDesc::BlocksSize() const { return GetBlockListBuilder().size(); }

void ProgramDesc::ClearBlocks() { GetMutableBlockListBuilder()->Clear(); }

template <>
proto::BlockDesc* ProgramDesc::GetBlock<proto::BlockDesc>(int32_t idx) {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return GetMutableBlockListBuilder()->GetMutable(idx);
}

template <>
proto::BlockDesc* ProgramDesc::AddBlock<proto::BlockDesc>() {
  return GetMutableBlockListBuilder()->New();
}

template <>
proto::OpVersionMap* ProgramDesc::GetOpVersionMap<proto::OpVersionMap>() {
  // op_version_map is not implemented on naive_buffer as
  // it's not useful in inference period.
  return nullptr;
}

int64_t ProgramDesc::Version() const {
  return desc_->GetField<Int64Builder>("version").data();
}

void ProgramDesc::SetVersion(int64_t version) {
  auto* builder = desc_->GetMutableField<Int64Builder>("version");
  CHECK(builder);
  builder->set(version);
}

const ListBuilder<proto::BlockDesc>& ProgramDesc::GetBlockListBuilder() const {
  return desc_->GetField<ListBuilder<proto::BlockDesc>>("blocks");
}

ListBuilder<proto::BlockDesc>* ProgramDesc::GetMutableBlockListBuilder() {
  auto* res = desc_->GetMutableField<ListBuilder<proto::BlockDesc>>("blocks");
  CHECK(res);
  return res;
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
