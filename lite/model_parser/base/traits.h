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

#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/utils/cp_logging.h"

#define LITE_MODEL_INTERFACE_NOT_IMPLEMENTED                \
  LOG(FATAL) << "This additional interface is temporarily " \
                "unavailable in flatbuffers read-only mode."

namespace paddle {
namespace lite {

// The AttrType is used to make the proto::AttrType portable.
enum class OpAttrType {
  INT = 0,
  FLOAT = 1,
  STRING = 2,
  INTS = 3,
  FLOATS = 4,
  STRINGS = 5,
  BOOLEAN = 6,
  BOOLEANS = 7,
  BLOCK = 8,
  LONG = 9,
  BLOCKS = 10,
  LONGS = 11,
  UNK,
};

enum class VarDataType {
  // Pod Types
  BOOL = 0,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  // Tensor<size_t> is used in C++.
  SIZE_T,
  UINT8,
  INT8,

  // Other types that may need additional descriptions
  LOD_TENSOR,
  SELECTED_ROWS,
  FEED_MINIBATCH,
  FETCH_LIST,
  STEP_SCOPES,
  LOD_RANK_TABLE,
  LOD_TENSOR_ARRAY,
  PLACE_LIST,
  READER,
  // Any runtime decided variable type is raw
  // raw variables should manage their own allocations
  // in operators like nccl_op
  RAW,
  TUPLE
};

inline VarDataType ConvertPrecisionType(lite_api::PrecisionType type) {
#define CASE(ptype, vtype)                \
  case lite_api::PrecisionType::k##ptype: \
    return lite::VarDataType::vtype;      \
    break
  switch (type) {
    CASE(Float, FP32);
    CASE(Int8, INT8);
    CASE(Int32, INT32);
    CASE(FP16, FP16);
    CASE(Bool, BOOL);
    CASE(Int64, INT64);
    CASE(Int16, INT16);
    default:
      LOG(FATAL) << "Illegal flatbuffer VarType.";
      return lite::VarDataType();
  }
#undef CASE
}

inline lite_api::PrecisionType ConvertPrecisionType(VarDataType type) {
#define CASE(ptype, vtype)                    \
  case lite::VarDataType::vtype:              \
    return lite_api::PrecisionType::k##ptype; \
    break
  switch (type) {
    CASE(Float, FP32);
    CASE(Int8, INT8);
    CASE(Int32, INT32);
    CASE(FP16, FP16);
    CASE(Bool, BOOL);
    CASE(Int64, INT64);
    CASE(Int16, INT16);
    default:
      LOG(FATAL) << "Illegal flatbuffer VarType.";
      return lite_api::PrecisionType();
  }
#undef CASE
}

struct Standard {};
struct Flatbuffers {};
struct Protobuf {};

template <typename T, typename U>
class VectorView;

template <typename T, typename U = Standard>
struct OpDataTypeTrait;

#define ATTR_TYPE_TRAIT_IMPL(T, type__)                \
  template <typename U>                                \
  struct OpDataTypeTrait<type__, U> {                  \
    typedef type__ ET;                                 \
    typedef type__ RT;                                 \
    static constexpr OpAttrType AT{OpAttrType::T};     \
    static constexpr const char* ATN{#T};              \
  };                                                   \
  template <typename U>                                \
  constexpr OpAttrType OpDataTypeTrait<type__, U>::AT; \
  template <typename U>                                \
  constexpr const char* OpDataTypeTrait<type__, U>::ATN;
#define ATTR_VECTOR_TYPE_TRAIT_IMPL(T, type__)                      \
  template <typename U>                                             \
  struct OpDataTypeTrait<std::vector<type__>, U> {                  \
    typedef type__ ET;                                              \
    typedef VectorView<type__, U> RT;                               \
    static constexpr OpAttrType AT{OpAttrType::T};                  \
    static constexpr const char* ATN{#T};                           \
  };                                                                \
  template <typename U>                                             \
  constexpr OpAttrType OpDataTypeTrait<std::vector<type__>, U>::AT; \
  template <typename U>                                             \
  constexpr const char* OpDataTypeTrait<std::vector<type__>, U>::ATN;

ATTR_TYPE_TRAIT_IMPL(BLOCK, int16_t);
ATTR_TYPE_TRAIT_IMPL(INT, int32_t);
ATTR_TYPE_TRAIT_IMPL(FLOAT, float);
ATTR_TYPE_TRAIT_IMPL(STRING, std::string);
ATTR_TYPE_TRAIT_IMPL(BOOLEAN, bool);
ATTR_TYPE_TRAIT_IMPL(LONG, int64_t);

ATTR_VECTOR_TYPE_TRAIT_IMPL(INTS, int32_t);
ATTR_VECTOR_TYPE_TRAIT_IMPL(FLOATS, float);
ATTR_VECTOR_TYPE_TRAIT_IMPL(STRINGS, std::string);
ATTR_VECTOR_TYPE_TRAIT_IMPL(LONGS, int64_t);

#undef ATTR_TYPE_TRAIT_IMPL
#undef ATTR_VECTOR_TYPE_TRAIT_IMPL

}  // namespace lite
}  // namespace paddle
