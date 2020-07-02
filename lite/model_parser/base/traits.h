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
#include "lite/model_parser/base/op_desc.h"

namespace paddle {
namespace lite {

struct Standard {};
struct Flatbuffers {};

template <OpAttrType Type, typename B = Standard>
struct OpAttrTypeTrait;

template <typename T>
struct OpDataTypeTrait;

#define TYPE_TRAIT_IMPL(T, type__)                  \
  template <>                                       \
  struct OpAttrTypeTrait<OpAttrType::T> {           \
    typedef type__ DT;                              \
  };                                                \
  template <>                                       \
  struct OpDataTypeTrait<type__> {                  \
    static constexpr OpAttrType AT = OpAttrType::T; \
    static constexpr const char* ATN = #T;          \
  };

TYPE_TRAIT_IMPL(INT, int32_t);
TYPE_TRAIT_IMPL(FLOAT, float);
TYPE_TRAIT_IMPL(STRING, std::string);
TYPE_TRAIT_IMPL(BOOLEAN, bool);
TYPE_TRAIT_IMPL(LONG, int64_t);
TYPE_TRAIT_IMPL(INTS, std::vector<int>);
TYPE_TRAIT_IMPL(FLOATS, std::vector<float>);
TYPE_TRAIT_IMPL(STRINGS, std::vector<std::string>);
TYPE_TRAIT_IMPL(LONGS, std::vector<int64_t>);
#undef TYPE_TRAIT_IMPL

}  // namespace lite
}  // namespace paddle
