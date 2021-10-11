/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/utils/fast_type_id.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <vector>

namespace paddle {
namespace lite {

// NOLINTNEXTLINE
#define PRIM_TYPES(A) \
  A(bool)             \
  A(int16_t)          \
  A(uint16_t)         \
  A(int)              \
  A(unsigned int)     \
  A(int64_t)          \
  A(uint64_t)         \
  A(float)            \
  A(double)           \
  A(long double)

TEST(FastTypeIdTest, PrimitiveTypes) {
  FastTypeIdType type_ids[] = {
#define A(T) FastTypeId<T>(),
      PRIM_TYPES(A)
#undef A
#define A(T) FastTypeId<const T>(),
          PRIM_TYPES(A)
#undef A
#define A(T) FastTypeId<volatile T>(),
              PRIM_TYPES(A)
#undef A
#define A(T) FastTypeId<const volatile T>(),
                  PRIM_TYPES(A)
#undef A
  };
  size_t total_type_ids = sizeof(type_ids) / sizeof(FastTypeIdType);

  for (int i = 0; i < total_type_ids; ++i) {
    EXPECT_EQ(type_ids[i], type_ids[i]);
    for (int j = 0; j < i; ++j) {
      EXPECT_NE(type_ids[i], type_ids[j]);
    }
  }
}

#define FIXED_WIDTH_TYPES(A) \
  A(int8_t)                  \
  A(uint8_t)                 \
  A(int16_t)                 \
  A(uint16_t)                \
  A(int32_t)                 \
  A(uint32_t)                \
  A(int64_t)                 \
  A(uint64_t)

TEST(FastTypeIdTest, FixedWidthTypes) {
  FastTypeIdType type_ids[] = {
#define A(T) FastTypeId<T>(),
      FIXED_WIDTH_TYPES(A)
#undef A
#define A(T) FastTypeId<const T>(),
          FIXED_WIDTH_TYPES(A)
#undef A
#define A(T) FastTypeId<volatile T>(),
              FIXED_WIDTH_TYPES(A)
#undef A
#define A(T) FastTypeId<const volatile T>(),
                  FIXED_WIDTH_TYPES(A)
#undef A
  };
  size_t total_type_ids = sizeof(type_ids) / sizeof(FastTypeIdType);

  for (int i = 0; i < total_type_ids; ++i) {
    EXPECT_EQ(type_ids[i], type_ids[i]);
    for (int j = 0; j < i; ++j) {
      EXPECT_NE(type_ids[i], type_ids[j]);
    }
  }
}

TEST(FastTypeIdTest, AliasTypes) {
  using int_alias = int;
  EXPECT_EQ(FastTypeId<int_alias>(), FastTypeId<int>());
}

TEST(FastTypeIdTest, TemplateSpecializations) {
  EXPECT_NE(FastTypeId<std::vector<int>>(), FastTypeId<std::vector<int64_t>>());

  EXPECT_NE((FastTypeId<std::map<int, float>>()),
            (FastTypeId<std::map<int, double>>()));
}

struct Base {};
struct Derived : Base {};
struct PDerived : private Base {};

TEST(FastTypeIdTest, Inheritance) {
  EXPECT_NE(FastTypeId<Base>(), FastTypeId<Derived>());
  EXPECT_NE(FastTypeId<Base>(), FastTypeId<PDerived>());
}

}  // namespace lite
}  // namespace paddle
