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

#include "lite/model_parser/cpp/block_desc.h"

namespace paddle {
namespace lite {
namespace cpp {

template <>
VarDesc* BlockDesc::GetVar<VarDesc>(int32_t idx) {
  CHECK_LT(idx, VarsSize()) << "idx >= vars.size()";
  return &vars_[idx];
}

template <>
VarDesc const* BlockDesc::GetVar<VarDesc>(int32_t idx) const {
  CHECK_LT(idx, VarsSize()) << "idx >= vars.size()";
  return &vars_[idx];
}

template <>
VarDesc* BlockDesc::AddVar<VarDesc>() {
  vars_.emplace_back();
  return &vars_.back();
}

template <>
OpDesc* BlockDesc::GetOp<OpDesc>(int32_t idx) {
  CHECK_LT(idx, OpsSize()) << "idx >= ops.size()";
  return &ops_[idx];
}

template <>
OpDesc const* BlockDesc::GetOp<OpDesc>(int32_t idx) const {
  CHECK_LT(idx, OpsSize()) << "idx >= ops.size()";
  return &ops_[idx];
}

template <>
OpDesc* BlockDesc::AddOp<OpDesc>() {
  ops_.emplace_back();
  return &ops_.back();
}

}  // namespace cpp
}  // namespace lite
}  // namespace paddle
