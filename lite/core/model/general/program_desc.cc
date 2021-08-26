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

#include "lite/core/model/general/program_desc.h"

namespace paddle {
namespace lite {
namespace general {

ProgramDesc::ProgramDesc(const ProgramDesc& other) { CopyFrom(other); }

ProgramDesc& ProgramDesc::operator=(const ProgramDesc& other) {
  CopyFrom(other);
  return *this;
}

void ProgramDesc::CopyFrom(const ProgramDesc& other) {
  version_ = other.Version();
  blocks_.clear();
  for (const auto& block : other.blocks()) {
    blocks_.emplace_back(new BlockDesc(*block));
  }
  if (other.HasOpVersionMap()) {
    op_version_map_.SetOpVersionMap(op_version_map_.GetOpVersionMap());
  }
}

template <>
BlockDesc* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= blocks.size()";
  return blocks_[idx].get();
}

template <>
BlockDesc const* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= blocks.size()";
  return blocks_[idx].get();
}

template <>
BlockDesc* ProgramDesc::AddBlock<BlockDesc>() {
  blocks_.emplace_back(new BlockDesc);
  return blocks_.back().get();
}

template <>
OpVersionMap* ProgramDesc::GetOpVersionMap<OpVersionMap>() {
  return &op_version_map_;
}

void ProgramDesc::SetOpVersionMap(
    std::map<std::string, int32_t> op_version_map) {
  op_version_map_.SetOpVersionMap(op_version_map);
}

}  // namespace general
}  // namespace lite
}  // namespace paddle
