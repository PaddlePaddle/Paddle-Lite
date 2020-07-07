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

struct Standard {};
struct Flatbuffers {};

template <typename T, typename U>
class VectorView;

template <typename T, typename U = Standard>
struct OpDataTypeTrait;

#define ATTR_TYPE_TRAIT_IMPL(T, type__)             \
  template <typename U>                             \
  struct OpDataTypeTrait<type__, U> {               \
    typedef type__ ET;                              \
    typedef type__ RT;                              \
    static constexpr OpAttrType AT = OpAttrType::T; \
    static constexpr const char* ATN = #T;          \
  };
#define ATTR_VECTOR_TYPE_TRAIT_IMPL(T, type__)      \
  template <typename U>                             \
  struct OpDataTypeTrait<std::vector<type__>, U> {  \
    typedef type__ ET;                              \
    typedef VectorView<type__, U> RT;               \
    static constexpr OpAttrType AT = OpAttrType::T; \
    static constexpr const char* ATN = #T;          \
  };

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
