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

#include "lite/model_parser/naive_buffer/param_desc.h"
#include <string>
#include <vector>
#include "lite/model_parser/naive_buffer/naive_buffer_wrapper_helper.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

std::string ParamDesc::Name() const {
  return desc_->GetField<StringBuilder>("name").data();
}

void ParamDesc::SetName(const std::string& name) {
  auto* build = desc_->GetMutableField<StringBuilder>("name");
  CHECK(build);
  build->set(name);
}

uint32_t ParamDesc::ModelVersion() const { return Version("model_version"); }

void ParamDesc::SetModelVersion(uint32_t version) {
  SetVersion("model_version", version);
}

uint32_t ParamDesc::TensorVersion() const { return Version("tensor_version"); }

void ParamDesc::SetTensorVersion(uint32_t version) {
  SetVersion("tensor_version", version);
}

uint64_t ParamDesc::LoDLevel() const {
  return desc_->GetField<UInt64Builder>("lod_level").data();
}

void ParamDesc::SetLoDLevel(uint64_t lod_level) {
  auto* build = desc_->GetMutableField<UInt64Builder>("lod_level");
  CHECK(build);
  build->set(lod_level);
}

std::vector<std::vector<uint64_t>> ParamDesc::LoD() const {
  using out_builder_type = ListBuilder<ListBuilder<UInt64Builder>>;

  std::vector<std::vector<uint64_t>> res;
  auto& out_builder = desc_->GetField<out_builder_type>("lod");
  for (size_t i = 0; i < out_builder.size(); ++i) {
    res.emplace_back(
        RepeatedToVector<uint64_t, UInt64Builder>(out_builder.Get(i)));
  }
  return res;
}

void ParamDesc::SetLoD(const std::vector<std::vector<uint64_t>>& lod) {
  using out_builder_type = ListBuilder<ListBuilder<UInt64Builder>>;

  auto* out_builder = desc_->GetMutableField<out_builder_type>("lod");
  CHECK(out_builder);
  out_builder->Clear();
  for (const auto& vals : lod) {
    VectorToRepeated<uint64_t, UInt64Builder>(vals, out_builder->New());
  }
}

VarDescAPI::VarDataType ParamDesc::GetDataType() const {
  using data_type_builder_t = EnumBuilder<proto::VarDataType>;

  auto data_type =
      GetTensorDesc().GetField<data_type_builder_t>("data_type").data();
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

void ParamDesc::SetDataType(VarDescAPI::VarDataType data_type) {
  using data_type_builder_t = EnumBuilder<proto::VarDataType>;

  auto* data_type_builder =
      GetMutableTensorDesc()->GetMutableField<data_type_builder_t>("data_type");
  CHECK(data_type_builder);
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
#undef SET_DATA_TYPE_CASE_ITEM
  }
}

std::vector<int64_t> ParamDesc::Dim() const {
  using out_builder_type = ListBuilder<Int64Builder>;

  auto& out_builder = GetTensorDesc().GetField<out_builder_type>("dims");
  return RepeatedToVector<int64_t, Int64Builder>(out_builder);
}

void ParamDesc::SetDim(const std::vector<int64_t>& dim) {
  using out_builder_type = ListBuilder<Int64Builder>;

  auto* out_builder =
      GetMutableTensorDesc()->GetMutableField<out_builder_type>("dims");
  CHECK(out_builder);
  VectorToRepeated<int64_t, Int64Builder>(dim, out_builder);
}

#define GET_DATA_IMPL(T, type__)                                            \
  template <>                                                               \
  std::vector<T> ParamDesc::Data() const {                                  \
    CHECK(GetDataType() == VarDescAPI::VarDataType::type__)                 \
        << "Data Type mismatch";                                            \
    std::vector<T> res;                                                     \
    auto& data_builder = desc_->GetField<PrimaryListBuilder<char>>("data"); \
    auto data = data_builder.data();                                        \
    size_t size = data_builder.size() / sizeof(T);                          \
    res.resize(size);                                                       \
    memcpy(&res[0], data, data_builder.size());                             \
    return res;                                                             \
  }

GET_DATA_IMPL(uint8_t, UINT8);
GET_DATA_IMPL(int8_t, INT8);
GET_DATA_IMPL(int16_t, INT16);
GET_DATA_IMPL(int32_t, INT32);
GET_DATA_IMPL(int64_t, INT64);
GET_DATA_IMPL(float, FP32);
GET_DATA_IMPL(double, FP64);
#undef GET_DATA_IMPL

// NOTE: Must set data type first
#define SET_DATA_COMMON_IMPL(T, type__, size__, data_ptr__)     \
  CHECK(GetDataType() == VarDescAPI::VarDataType::type__)       \
      << "Data Type mismatch, call SetDataType first.";         \
  auto* data_builder =                                          \
      desc_->GetMutableField<PrimaryListBuilder<char>>("data"); \
  CHECK(data_builder);                                          \
  data_builder->Clear();                                        \
  size_t size = size__ * sizeof(T);                             \
  auto* data_ptr = reinterpret_cast<const char*>(data_ptr__);   \
  data_builder->set(data_ptr, size);

#define SET_DATA_IMPL(T, type__)                                \
  template <>                                                   \
  void ParamDesc::SetData<T>(const std::vector<T>& data) {      \
    SET_DATA_COMMON_IMPL(T, type__, data.size(), &data[0])      \
  }                                                             \
                                                                \
  template <>                                                   \
  void ParamDesc::SetData<T>(const T* data, size_t data_size) { \
    CHECK(data);                                                \
    SET_DATA_COMMON_IMPL(T, type__, data_size, data);           \
  }

SET_DATA_IMPL(uint8_t, UINT8);
SET_DATA_IMPL(int8_t, INT8);
SET_DATA_IMPL(int16_t, INT16);
SET_DATA_IMPL(int32_t, INT32);
SET_DATA_IMPL(int64_t, INT64);
SET_DATA_IMPL(float, FP32);
SET_DATA_IMPL(double, FP64);
#undef SET_DATA_IMPL
#undef SET_DATA_COMMON_IMPL

uint32_t ParamDesc::Version(const std::string& name) const {
  auto& builder = desc_->GetField<UInt32Builder>(name);
  return builder.data();
}

void ParamDesc::SetVersion(const std::string& name, uint32_t version) {
  auto* builder = desc_->GetMutableField<UInt32Builder>(name);
  CHECK(builder);
  return builder->set(version);
}

const proto::TensorDesc& ParamDesc::GetTensorDesc() const {
  return desc_->GetField<proto::TensorDesc>("tensor_desc");
}

proto::TensorDesc* ParamDesc::GetMutableTensorDesc() {
  auto* builder = desc_->GetMutableField<proto::TensorDesc>("tensor_desc");
  CHECK(builder);
  return builder;
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
