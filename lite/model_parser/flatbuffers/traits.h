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

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/model_parser/base/traits.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

inline lite::VarDataType ConvertVarType(proto::VarType_::Type type) {
#define CASE(type)                   \
  case proto::VarType_::Type_##type: \
    return lite::VarDataType::type;  \
    break
  switch (type) {
    CASE(BOOL);
    CASE(INT16);
    CASE(INT32);
    CASE(INT64);
    CASE(FP16);
    CASE(FP32);
    CASE(FP64);
    CASE(LOD_TENSOR);
    CASE(SELECTED_ROWS);
    CASE(FEED_MINIBATCH);
    CASE(FETCH_LIST);
    CASE(STEP_SCOPES);
    CASE(LOD_RANK_TABLE);
    CASE(LOD_TENSOR_ARRAY);
    CASE(PLACE_LIST);
    CASE(READER);
    CASE(RAW);
    CASE(TUPLE);
    CASE(SIZE_T);
    CASE(UINT8);
    CASE(INT8);
    default:
      LOG(FATAL) << "Illegal flatbuffer VarType.";
      return lite::VarDataType();
  }
#undef CASE
}

inline proto::VarType_::Type ConvertVarType(lite::VarDataType type) {
#define CASE(type)                       \
  case lite::VarDataType::type:          \
    return proto::VarType_::Type_##type; \
    break
  switch (type) {
    CASE(BOOL);
    CASE(INT16);
    CASE(INT32);
    CASE(INT64);
    CASE(FP16);
    CASE(FP32);
    CASE(FP64);
    CASE(LOD_TENSOR);
    CASE(SELECTED_ROWS);
    CASE(FEED_MINIBATCH);
    CASE(FETCH_LIST);
    CASE(STEP_SCOPES);
    CASE(LOD_RANK_TABLE);
    CASE(LOD_TENSOR_ARRAY);
    CASE(PLACE_LIST);
    CASE(READER);
    CASE(RAW);
    CASE(TUPLE);
    CASE(SIZE_T);
    CASE(UINT8);
    CASE(INT8);
    default:
      LOG(FATAL) << "Illegal flatbuffer VarType.";
      return proto::VarType_::Type();
  }
#undef CASE
}

inline lite::OpAttrType ConvertAttrType(proto::AttrType type) {
#define CASE(type)                 \
  case proto::AttrType_##type:     \
    return lite::OpAttrType::type; \
    break
  switch (type) {
    CASE(INT);
    CASE(FLOAT);
    CASE(STRING);
    CASE(INTS);
    CASE(FLOATS);
    CASE(STRINGS);
    CASE(BOOLEAN);
    CASE(BOOLEANS);
    CASE(BLOCK);
    CASE(LONG);
    CASE(BLOCKS);
    CASE(LONGS);
    default:
      LOG(FATAL) << "Illegal flatbuffer AttrType.";
      return lite::OpAttrType();
  }
#undef CASE
}

inline proto::AttrType ConvertAttrType(lite::OpAttrType type) {
#define CASE(type)                 \
  case lite::OpAttrType::type:     \
    return proto::AttrType_##type; \
    break
  switch (type) {
    CASE(INT);
    CASE(FLOAT);
    CASE(STRING);
    CASE(INTS);
    CASE(FLOATS);
    CASE(STRINGS);
    CASE(BOOLEAN);
    CASE(BOOLEANS);
    CASE(BLOCK);
    CASE(LONG);
    CASE(BLOCKS);
    CASE(LONGS);
    default:
      LOG(FATAL) << "Illegal flatbuffer AttrType.";
      return proto::AttrType();
  }
#undef CASE
}

template <typename FlatbuffersMapT, typename KeyT = std::string>
KeyT GetKey(const std::unique_ptr<FlatbuffersMapT>& object);

template <typename FlatbuffersMapT, typename KeyT = std::string>
void SetKey(const KeyT& key, std::unique_ptr<FlatbuffersMapT>* object);

#define GET_KEY_INSTANCE(type, key, key_type)                             \
  template <>                                                             \
  inline key_type GetKey<proto::type>(                                    \
      const std::unique_ptr<proto::type>& object) {                       \
    return object->key;                                                   \
  }                                                                       \
  template <>                                                             \
  inline void SetKey<proto::type>(const key_type& key_in,                 \
                                  std::unique_ptr<proto::type>* object) { \
    (*object)->key = key_in;                                              \
  }
GET_KEY_INSTANCE(OpDesc_::VarT, parameter, std::string);
GET_KEY_INSTANCE(OpDesc_::AttrT, name, std::string);
#undef GET_KEY_INSTANCE

template <typename MapT, typename KeyT = std::string>
struct CompareLessThanKey {
  bool operator()(const std::unique_ptr<MapT>& lhs, const KeyT& rhs) {
    return GetKey(lhs) < rhs;
  }
  bool operator()(const KeyT& lhs, const std::unique_ptr<MapT>& rhs) {
    return lhs < GetKey(rhs);
  }
};

template <typename MapT>
struct CompareLessThan {
  bool operator()(const std::unique_ptr<MapT>& lhs,
                  const std::unique_ptr<MapT>& rhs) {
    return GetKey(lhs) < GetKey(rhs);
  }
};

template <typename MapT,
          typename KeyT = std::string,
          typename CompareFunc = CompareLessThanKey<MapT, KeyT>>
typename std::vector<std::unique_ptr<MapT>>::const_iterator GetKeyIterator(
    const KeyT& key, const std::vector<std::unique_ptr<MapT>>& vector) {
  auto iter =
      std::lower_bound(vector.begin(), vector.end(), key, CompareFunc());
  CHECK_EQ(GetKey(*iter), key);
  return iter;
}

template <typename MapT,
          typename KeyT = std::string,
          typename CompareFunc = CompareLessThanKey<MapT, KeyT>>
typename std::vector<std::unique_ptr<MapT>>::iterator InsertPair(
    const KeyT& key,
    std::unique_ptr<MapT>&& val,
    std::vector<std::unique_ptr<MapT>>* vector) {
  auto iter =
      std::lower_bound(vector->begin(), vector->end(), key, CompareFunc());
  return vector->insert(iter, std::forward<std::unique_ptr<MapT>>(val));
}

template <typename MapT,
          typename KeyT = std::string,
          typename CompareFunc = CompareLessThanKey<MapT, KeyT>>
bool HasKey(const KeyT& key, const std::vector<std::unique_ptr<MapT>>& vector) {
  return std::binary_search(vector.begin(), vector.end(), key, CompareFunc());
}

template <typename MapT, typename CompareFunc = CompareLessThan<MapT>>
void Sort(std::vector<std::unique_ptr<MapT>>* vector) {
  std::sort(vector->begin(), vector->end(), CompareFunc());
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
