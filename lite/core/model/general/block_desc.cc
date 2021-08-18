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

#include "lite/core/model/general/block_desc.h"

namespace paddle {
namespace lite {
namespace general {

template <>
VarDesc* BlockDesc::GetVar<VarDesc>(int32_t idx) {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return vars_[idx].get();
}

template <>
VarDesc const* BlockDesc::GetVar<VarDesc>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return vars_[idx].get();
}

template <>
VarDesc* BlockDesc::AddVar<VarDesc>() {
  vars_.emplace_back(new VarDesc);
  return vars_.back().get();
}

template <>
OpDesc* BlockDesc::GetOp<OpDesc>(int32_t idx) {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= ops.size()";
  return ops_[idx].get();
}

template <>
OpDesc const* BlockDesc::GetOp<OpDesc>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= ops.size()";
  return ops_[idx].get();
}

template <>
OpDesc* BlockDesc::AddOp<OpDesc>() {
  ops_.emplace_back(new OpDesc);
  return ops_.back().get();
}

void BlockDesc::CopyFrom(const BlockDesc& desc) {
  ops_.clear();
  vars_.clear();
  SetIdx(desc.Idx());
  SetParentIdx(desc.ParentIdx());
  SetForwardBlockIdx(desc.ForwardBlockIdx());
  for (size_t i = 0; i < desc.OpsSize(); ++i) {
    ops_.emplace_back(new OpDesc(*desc.GetOp<OpDesc>(i)));
  }
  for (size_t i = 0; i < desc.VarsSize(); ++i) {
    vars_.emplace_back(new VarDesc(*desc.GetVar<VarDesc>(i)));
  }
}

BlockDesc::BlockDesc(const BlockDesc& desc) { CopyFrom(desc); }

BlockDesc& BlockDesc::operator=(const BlockDesc& desc) {
  CopyFrom(desc);
  return *this;
}

}  // namespace general
}  // namespace lite
}  // namespace paddle
