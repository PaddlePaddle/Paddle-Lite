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

#include "lite/model_parser/flatbuffers/program_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

template <>
proto::BlockDesc const* ProgramDescView::GetBlock<proto::BlockDesc>(
    int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= blocks.size()";
  return desc_->blocks()->Get(idx);
}

template <>
BlockDescView const* ProgramDescView::GetBlock<BlockDescView>(
    int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= blocks.size()";
  return &blocks_[idx];
}

template <>
proto::BlockDescT* ProgramDesc::GetBlock<proto::BlockDescT>(int32_t idx) {
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= vars.size()";
  return blocks_[idx]->raw_desc();
}

template <>
proto::BlockDescT* ProgramDesc::AddBlock<proto::BlockDescT>() {
  desc_.blocks.push_back(
      std::unique_ptr<proto::BlockDescT>(new proto::BlockDescT));
  SyncBlocks();
  return blocks_.back()->raw_desc();
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
