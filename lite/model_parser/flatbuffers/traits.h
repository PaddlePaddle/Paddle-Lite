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

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
