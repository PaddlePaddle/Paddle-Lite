// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/sve/softmax_sve.h"

#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
typedef __fp16 float16_t;
#endif
#include <arm_sve.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve {

template <size_t element_size>
inline svbool_t svptrue_size();

template <>
inline svbool_t svptrue_size<64>() {
  return svptrue_b64();
}

template <>
inline svbool_t svptrue_size<32>() {
  return svptrue_b32();
}

template <>
inline svbool_t svptrue_size<16>() {
  return svptrue_b16();
}

template <>
inline svbool_t svptrue_size<8>() {
  return svptrue_b8();
}

template <typename T>
svbool_t svptrue() {
  return svptrue_size<sizeof(T) * 8>();
}

template <size_t element_size>
inline uint64_t svcnt_size();

template <>
inline uint64_t svcnt_size<64>() {
  return svcntd();
}

template <>
inline uint64_t svcnt_size<32>() {
  return svcntw();
}

template <>
inline uint64_t svcnt_size<16>() {
  return svcnth();
}

template <>
inline uint64_t svcnt_size<8>() {
  return svcntb();
}

template <typename T>
inline uint64_t svcnt() {
  return svcnt_size<sizeof(T) * 8>();
}

#define SVDUP_N_IMPL(Intype, Vectortype, postfix) \
  inline Vectortype svdup_n(Intype a) { return svdup_n_##postfix(a); }

SVDUP_N_IMPL(int8_t, svint8_t, s8)
SVDUP_N_IMPL(int16_t, svint16_t, s16)
SVDUP_N_IMPL(int32_t, svint32_t, s32)
SVDUP_N_IMPL(int64_t, svint64_t, s64)
SVDUP_N_IMPL(uint8_t, svuint8_t, u8)
SVDUP_N_IMPL(uint16_t, svuint16_t, u16)
SVDUP_N_IMPL(uint32_t, svuint32_t, u32)
SVDUP_N_IMPL(uint64_t, svuint64_t, u64)
SVDUP_N_IMPL(float16_t, svfloat16_t, f16)
SVDUP_N_IMPL(float, svfloat32_t, f32)
SVDUP_N_IMPL(bfloat16_t, svbfloat16_t, bf16)

#undef SVDUP_N_IMPL

#define SVWHILELT_IMPL(type)                           \
  template <size_t element_size>                       \
  inline svbool_t svwhilelt_size(type a, type b);      \
  template <>                                          \
  inline svbool_t svwhilelt_size<64>(type a, type b) { \
    return svwhilelt_b64(a, b);                        \
  }                                                    \
  template <>                                          \
  inline svbool_t svwhilelt_size<32>(type a, type b) { \
    return svwhilelt_b32(a, b);                        \
  }                                                    \
  template <>                                          \
  inline svbool_t svwhilelt_size<16>(type a, type b) { \
    return svwhilelt_b16(a, b);                        \
  }                                                    \
  template <>                                          \
  inline svbool_t svwhilelt_size<8>(type a, type b) {  \
    return svwhilelt_b8(a, b);                         \
  }

SVWHILELT_IMPL(int32_t)
SVWHILELT_IMPL(int64_t)

#undef SVWHILELT_IMPL

template <typename Dtype, typename IndexType>
inline svbool_t svwhilelt(IndexType a, IndexType b) {
  return svwhilelt_size<sizeof(Dtype) * 8>(a, b);
}

#define SVEXP_IMPL(vtype, postfix)                    \
  inline vtype svexp_z(svbool_t pg, const vtype &a) { \
    return svexp_##postfix##_z(pg, a);                \
  }

SVEXP_IMPL(svfloat32_t, f32)
SVEXP_IMPL(svfloat16_t, f16)

#undef SVEXP_IMPL
}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
