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

VarDescAPI::VarDataType VarDesc::GetDataType() const {
  using data_type_builder_t = EnumBuilder<proto::VarDataType>;

  auto data_type = desc_->GetField<proto::TensorDesc>("tensor_desc")
                       .GetField<data_type_builder_t>("data_type")
                       .data();
#define GET_DATA_TYPE_CASE_ITEM(type__) \
  case proto::VarDataType::type__:      \
    return VarDescAPI::VarDataType::type__

  switch (data_type) {
    // Only support primary data type now.
    GET_DATA_TYPE_CASE_ITEM(UINT8);
    GET_DATA_TYPE_CASE_ITEM(INT8);
    GET_DATA_TYPE_CASE_ITEM(INT16);
    GET_DATA_TYPE_CASE_ITEM(INT32);
    GET_DATA_TYPE_CASE_ITEM(INT64);
    GET_DATA_TYPE_CASE_ITEM(FP32);
    GET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      LOG(FATAL) << "Unknown var data type";
  }
  return VarDescAPI::VarDataType();
#undef GET_DATA_TYPE_CASE_ITEM
}

proto::VarType* VarDesc::GetMutableVarType() {
  auto* builder = desc_->GetMutableField<proto::VarType>("type");
  CHECK(builder);
  return builder;
}

// todo : SetDataType function is commented out temporarily
// because of Compatibility issues. The Compatibility issue
// should be fixed later and the code below should be applied
// later. @DannyIsFunny
void VarDesc::SetDataType(VarDescAPI::VarDataType data_type) {
  /*  using data_type_builder_t = EnumBuilder<proto::VarDataType>;
    auto data_type_builder =
        desc_->GetMutableField<proto::TensorDesc>("tensor_desc")
            ->GetMutableField<data_type_builder_t>("data_type");
  #define SET_DATA_TYPE_CASE_ITEM(type__)                 \
    case VarDescAPI::VarDataType::type__:                 \
      data_type_builder->set(proto::VarDataType::type__); \
      break

    switch (data_type) {
      // Only support primary data type now.
      SET_DATA_TYPE_CASE_ITEM(UINT8);
      SET_DATA_TYPE_CASE_ITEM(INT8);
      SET_DATA_TYPE_CASE_ITEM(INT16);
      SET_DATA_TYPE_CASE_ITEM(INT32);
      SET_DATA_TYPE_CASE_ITEM(INT64);
      SET_DATA_TYPE_CASE_ITEM(FP32);
      SET_DATA_TYPE_CASE_ITEM(FP64);
      default:
        LOG(FATAL) << "Unknown var data type";
    }
  #undef SET_DATA_TYPE_CASE_ITEM
  */
}

// Get var's shape
std::vector<int64_t> VarDesc::GetShape() const {
  using data_type_builder_t = ListBuilder<Int64Builder>;
  auto out_builder = desc_->GetField<proto::TensorDesc>("tensor_desc")
                         .GetField<data_type_builder_t>("dims");
  return RepeatedToVector<int64_t, Int64Builder>(out_builder);
}

// Set var's shape
// todo : SetDataType function is commented out temporarily
// because of Compatibility issues. The Compatibility issue
// should be fixed later and the code below should be applied
// later. @DannyIsFunny
void VarDesc::SetShape(const std::vector<int64_t>& dims) {
  /*  using out_builder_type = ListBuilder<Int64Builder>;
    auto out_builder = desc_->GetMutableField<proto::TensorDesc>("tensor_desc")
                           ->GetMutableField<out_builder_type>("dims");
    CHECK(out_builder);
    VectorToRepeated<int64_t, Int64Builder>(dims, out_builder);*/
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
