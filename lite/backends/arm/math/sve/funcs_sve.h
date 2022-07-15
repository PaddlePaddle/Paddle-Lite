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
/*
 * The following function is base on
 * https://github.com/ARM-software/ComputeLibrary/
 *
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <arm_neon.h>
#include <arm_sve.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/sve/gemm_sve.h"
#include "lite/backends/arm/math/sve/gemm_sve_i8mm.h"
#include "lite/backends/arm/math/sve/softmax_sve.h"

#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
typedef __fp16 float16_t;
#endif

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
#ifdef ENABLE_ARM_FP16
SVDUP_N_IMPL(float16_t, svfloat16_t, f16)
#endif
SVDUP_N_IMPL(float, svfloat32_t, f32)

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

inline svfloat32_t svtaylor_poly_f32_z(svbool_t pg,
                                       svfloat32_t x,
                                       svfloat32_t coeff_1,
                                       svfloat32_t coeff_2,
                                       svfloat32_t coeff_3,
                                       svfloat32_t coeff_4,
                                       svfloat32_t coeff_5,
                                       svfloat32_t coeff_6,
                                       svfloat32_t coeff_7,
                                       svfloat32_t coeff_8) {
  const auto A = svmla_f32_z(pg, coeff_1, coeff_5, x);
  const auto B = svmla_f32_z(pg, coeff_3, coeff_7, x);
  const auto C = svmla_f32_z(pg, coeff_2, coeff_6, x);
  const auto D = svmla_f32_z(pg, coeff_4, coeff_8, x);
  const auto x2 = svmul_f32_z(pg, x, x);
  const auto x4 = svmul_f32_z(pg, x2, x2);
  const auto res =
      svmla_f32_z(pg, svmla_f32_z(pg, A, B, x2), svmla_f32_z(pg, C, D, x2), x4);
  return res;
}

inline svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t x) {
  const auto CONST_LN2 = svdup_n_f32(0.6931471805f);      // ln(2)
  const auto CONST_INV_LN2 = svdup_n_f32(1.4426950408f);  // 1/ln(2)
  const auto CONST_INF = svdup_n_f32(std::numeric_limits<float>::infinity());
  const auto CONST_MAX_INPUT = svdup_n_f32(88.7f);
  const auto CONST_0 = svdup_n_f32(0.f);
  const auto CONST_NEGATIVE_126 = svdup_n_s32(-126);

  /** Exponent polynomial coefficients */
  const svfloat32_t exp_tab_1 = svdup_n_f32(1.f);
  const svfloat32_t exp_tab_2 = svdup_n_f32(0.0416598916054f);
  const svfloat32_t exp_tab_3 = svdup_n_f32(0.500000596046f);
  const svfloat32_t exp_tab_4 = svdup_n_f32(0.0014122662833f);
  const svfloat32_t exp_tab_5 = svdup_n_f32(1.00000011921f);
  const svfloat32_t exp_tab_6 = svdup_n_f32(0.00833693705499f);
  const svfloat32_t exp_tab_7 = svdup_n_f32(0.166665703058f);
  const svfloat32_t exp_tab_8 = svdup_n_f32(0.000195780929062f);

  // Perform range reduction [-log(2),log(2)]
  auto m = svcvt_s32_f32_z(pg, svmul_f32_z(pg, x, CONST_INV_LN2));
  auto val = svmls_f32_z(pg, x, svcvt_f32_s32_z(pg, m), CONST_LN2);

  // Polynomial Approximation
  auto poly = svtaylor_poly_f32_z(pg,
                                  val,
                                  exp_tab_1,
                                  exp_tab_2,
                                  exp_tab_3,
                                  exp_tab_4,
                                  exp_tab_5,
                                  exp_tab_6,
                                  exp_tab_7,
                                  exp_tab_8);

  // Reconstruct
  poly = svreinterpret_f32_s32(
      svqadd_s32(svreinterpret_s32_f32(poly), svlsl_n_s32_z(pg, m, 23)));

  // Handle underflow
  svbool_t ltpg = svcmplt_s32(pg, m, CONST_NEGATIVE_126);
  poly = svsel_f32(ltpg, CONST_0, poly);

  // Handle overflow
  svbool_t gtpg = svcmpgt_f32(pg, x, CONST_MAX_INPUT);
  poly = svsel_f32(gtpg, CONST_INF, poly);

  return poly;
}

#ifdef ENABLE_ARM_FP16
inline svfloat16_t svtaylor_poly_f16_z(svbool_t pg,
                                       svfloat16_t x,
                                       svfloat16_t coeff_1,
                                       svfloat16_t coeff_2,
                                       svfloat16_t coeff_3,
                                       svfloat16_t coeff_4,
                                       svfloat16_t coeff_5,
                                       svfloat16_t coeff_6,
                                       svfloat16_t coeff_7,
                                       svfloat16_t coeff_8) {
  const auto A = svmla_f16_z(pg, coeff_1, coeff_5, x);
  const auto B = svmla_f16_z(pg, coeff_3, coeff_7, x);
  const auto C = svmla_f16_z(pg, coeff_2, coeff_6, x);
  const auto D = svmla_f16_z(pg, coeff_4, coeff_8, x);
  const auto x2 = svmul_f16_z(pg, x, x);
  const auto x4 = svmul_f16_z(pg, x2, x2);
  const auto res =
      svmla_f16_z(pg, svmla_f16_z(pg, A, B, x2), svmla_f16_z(pg, C, D, x2), x4);
  return res;
}

inline svfloat16_t svexp_f16_z(svbool_t pg, svfloat16_t x) {
  auto bottom = svcvt_f32_z(pg, x);
#if defined(LITE_WITH_ARM8_SVE2)
  auto top = svcvtlt_f32_x(pg, x);
  auto pg_top = pg;
#else
  auto pg_top = svptrue_b16();
  auto top = svcvt_f32_z(
      pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(x))));
#endif
  bottom = svexp_f32_z(pg, bottom);
  top = svexp_f32_z(pg_top, top);

#if defined(LITE_WITH_ARM8_SVE2)
  return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else
  return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif
}
#endif

template <typename Dtype, typename IndexType>
inline svbool_t svwhilelt(IndexType a, IndexType b) {
  return svwhilelt_size<sizeof(Dtype) * 8>(a, b);
}

#define SVEXP_IMPL(vtype, postfix)                    \
  inline vtype svexp_z(svbool_t pg, const vtype &a) { \
    return svexp_##postfix##_z(pg, a);                \
  }

SVEXP_IMPL(svfloat32_t, f32)
#ifdef ENABLE_ARM_FP16
SVEXP_IMPL(svfloat16_t, f16)
#endif

#undef SVEXP_IMPL
}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
