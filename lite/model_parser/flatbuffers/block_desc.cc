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

#include "lite/model_parser/flatbuffers/block_desc.h"
#include <memory>

namespace paddle {
namespace lite {
namespace fbs {

template <>
proto::VarDesc const* BlockDescView::GetVar<proto::VarDesc>(int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return desc_->vars()->Get(idx);
}

template <>
proto::OpDesc const* BlockDescView::GetOp<proto::OpDesc>(int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= ops.size()";
  return desc_->ops()->Get(idx);
}

template <>
VarDescView const* BlockDescView::GetVar<VarDescView>(int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return &vars_[idx];
}

template <>
OpDescView const* BlockDescView::GetOp<OpDescView>(int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= ops.size()";
  return &ops_[idx];
}

#ifdef LITE_WITH_FLATBUFFERS_DESC
template <>
proto::VarDescT* BlockDesc::GetVar<proto::VarDescT>(int32_t idx) {
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return vars_[idx]->raw_desc();
}

template <>
proto::VarDescT* BlockDesc::AddVar<proto::VarDescT>() {
  desc_->vars.push_back(std::unique_ptr<proto::VarDescT>(new proto::VarDescT));
  SyncVars();
  return vars_.back()->raw_desc();
}

template <>
proto::OpDescT* BlockDesc::GetOp<proto::OpDescT>(int32_t idx) {
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= vars.size()";
  return ops_[idx]->raw_desc();
}

template <>
proto::OpDescT* BlockDesc::AddOp<proto::OpDescT>() {
  desc_->ops.push_back(std::unique_ptr<proto::OpDescT>(new proto::OpDescT));
  SyncOps();
  return ops_.back()->raw_desc();
}
#endif  // LITE_WITH_FLATBUFFERS_DESC

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
