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

#include "lite/model_parser/naive_buffer/var_desc.h"
#include <string>

namespace paddle {
namespace lite {
namespace naive_buffer {

std::string VarDesc::Name() const {
  auto& builder = desc_->GetField<StringBuilder>("name");
  return builder.data();
}

void VarDesc::SetName(std::string name) {
  auto* builder = desc_->GetMutableField<StringBuilder>("name");
  CHECK(builder);
  return builder->set(name);
}

VarDescAPI::Type VarDesc::GetType() const {
  using PbType = proto::VarDataType;
  using type_builder_t = EnumBuilder<PbType>;

  auto type = GetVarType().GetField<type_builder_t>("type").data();
#define GET_TYPE_CASE_ITEM(type__) \
  case PbType::type__:             \
    return VarDescAPI::Type::type__

  switch (type) {
    GET_TYPE_CASE_ITEM(LOD_TENSOR);
    GET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    GET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    GET_TYPE_CASE_ITEM(SELECTED_ROWS);
    GET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    GET_TYPE_CASE_ITEM(FETCH_LIST);
    GET_TYPE_CASE_ITEM(STEP_SCOPES);
    GET_TYPE_CASE_ITEM(PLACE_LIST);
    GET_TYPE_CASE_ITEM(READER);
    default:
      LOG(FATAL) << "Unknown var type";
      return VarDescAPI::Type();
  }
#undef GET_TYPE_CASE_ITEM
}

void VarDesc::SetType(VarDescAPI::Type type) {
  using PbType = proto::VarDataType;
  using type_builder_t = EnumBuilder<PbType>;

  auto* type_builder =
      GetMutableVarType()->GetMutableField<type_builder_t>("type");
  CHECK(type_builder);
#define SET_TYPE_CASE_ITEM(type__)     \
  case VarDescAPI::Type::type__:       \
    type_builder->set(PbType::type__); \
    break

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      LOG(FATAL) << "Unknown var type";
  }
#undef SET_TYPE_CASE_ITEM
}

bool VarDesc::Persistable() const {
  auto& builder = desc_->GetField<BoolBuilder>("persistable");
  return builder.data();
}

void VarDesc::SetPersistable(bool persistable) {
  auto* builder = desc_->GetMutableField<BoolBuilder>("persistable");
  CHECK(builder);
  return builder->set(persistable);
}

const proto::VarType& VarDesc::GetVarType() const {
  return desc_->GetField<proto::VarType>("type");
}

proto::VarType* VarDesc::GetMutableVarType() {
  auto* builder = desc_->GetMutableField<proto::VarType>("type");
  CHECK(builder);
  return builder;
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
