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

#include "lite/model_parser/naive_buffer/block_desc.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

int32_t BlockDesc::Idx() const {
  return desc_->GetField<Int32Builder>("idx").data();
}

void BlockDesc::SetIdx(int32_t idx) {
  auto* builder = desc_->GetMutableField<Int32Builder>("idx");
  CHECK(builder);
  builder->set(idx);
}

int32_t BlockDesc::ParentIdx() const {
  return desc_->GetField<Int32Builder>("parent_idx").data();
}

void BlockDesc::SetParentIdx(int32_t idx) {
  auto* builder = desc_->GetMutableField<Int32Builder>("parent_idx");
  CHECK(builder);
  builder->set(idx);
}

size_t BlockDesc::VarsSize() const { return GetVarListBuilder().size(); }

void BlockDesc::ClearVars() { GetMutableVarListBuilder()->Clear(); }

template <>
proto::VarDesc* BlockDesc::GetVar<proto::VarDesc>(int32_t idx) {
  CHECK_LT(idx, VarsSize()) << "idx >= vars.size()";
  return GetMutableVarListBuilder()->GetMutable(idx);
}

template <>
proto::VarDesc* BlockDesc::AddVar<proto::VarDesc>() {
  return GetMutableVarListBuilder()->New();
}

size_t BlockDesc::OpsSize() const { return GetOpListBuilder().size(); }

void BlockDesc::ClearOps() { return GetMutableOpListBuilder()->Clear(); }

template <>
proto::OpDesc* BlockDesc::GetOp<proto::OpDesc>(int32_t idx) {
  CHECK_LT(idx, OpsSize()) << "idx >= ops.size()";
  return GetMutableOpListBuilder()->GetMutable(idx);
}

template <>
proto::OpDesc* BlockDesc::AddOp<proto::OpDesc>() {
  return GetMutableOpListBuilder()->New();
}

int32_t BlockDesc::ForwardBlockIdx() const {
  return desc_->GetField<Int32Builder>("forward_block_idx").data();
}

void BlockDesc::SetForwardBlockIdx(int32_t idx) {
  auto* builder = desc_->GetMutableField<Int32Builder>("forward_block_idx");
  CHECK(builder);
  builder->set(idx);
}

const ListBuilder<proto::VarDesc>& BlockDesc::GetVarListBuilder() const {
  return desc_->GetField<ListBuilder<proto::VarDesc>>("vars");
}

ListBuilder<proto::VarDesc>* BlockDesc::GetMutableVarListBuilder() {
  auto* res = desc_->GetMutableField<ListBuilder<proto::VarDesc>>("vars");
  CHECK(res);
  return res;
}

const ListBuilder<proto::OpDesc>& BlockDesc::GetOpListBuilder() const {
  return desc_->GetField<ListBuilder<proto::OpDesc>>("ops");
}

ListBuilder<proto::OpDesc>* BlockDesc::GetMutableOpListBuilder() {
  auto* res = desc_->GetMutableField<ListBuilder<proto::OpDesc>>("ops");
  CHECK(res);
  return res;
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
