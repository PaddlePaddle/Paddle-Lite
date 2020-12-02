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

#include "lite/model_parser/general/program_desc.h"

namespace paddle {
namespace lite {
namespace general {

template <>
BlockDesc* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return &blocks_[idx];
}

template <>
BlockDesc const* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) const {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return &blocks_[idx];
}

template <>
BlockDesc* ProgramDesc::AddBlock<BlockDesc>() {
  blocks_.emplace_back();
  return &blocks_.back();
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
