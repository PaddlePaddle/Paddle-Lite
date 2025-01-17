/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __loongarch_asx

#include "lite/backends/loongarch/math/activation_functions.h"
#include "lite/backends/loongarch/math/include/mathfuns.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
namespace detail {

namespace forward {
namespace lasx {
__m256 Relu(const __m256 a) {
  __m256 tmp = lasx_set1_f32(0.0f);
  return lasx_max_f32(a, tmp);
}

__m256 Sigmoid(const __m256 a) {
  __m256 max = lasx_set1_f32(SIGMOID_THRESHOLD_MAX);
  __m256 min = lasx_set1_f32(SIGMOID_THRESHOLD_MIN);
  __m256 tmp = lasx_max_f32(a, min);
  tmp = lasx_min_f32(tmp, max);
  tmp = lasx_sub_f32(lasx_set1_f32(0.0f), tmp);
  tmp = lite::loongarch::math::exp256_ps(tmp);
  tmp = lasx_add_f32(lasx_set1_f32(1.0f), tmp);
  tmp = lasx_div_f32(lasx_set1_f32(1.0f), tmp);
  return tmp;
}

__m256 Tanh(const __m256 a) {
  __m256 max = lasx_set1_f32(EXP_MAX_INPUT);
  __m256 tmp = lasx_mul_f32(lasx_set1_f32(-2.0f), a);
  tmp = lasx_min_f32(tmp, max);
  tmp = lite::loongarch::math::exp256_ps(tmp);
  return lasx_sub_f32(lasx_div_f32(lasx_set1_f32(2.0f),
                                     lasx_add_f32(lasx_set1_f32(1.0f), tmp)),
                       lasx_set1_f32(1.0f));
}

__m256 Identity(const __m256 a) { return a; }

}  // namespace lasx
}  // namespace forward

namespace backward {
namespace lasx {
__m256 Relu(const __m256 a, const __m256 b) {
  return lasx_mul_f32(
      a,
      lasx_and_f32(lasx_xvfcmp_slt_s(lasx_set1_f32(0.0f), b),
                    lasx_set1_f32(1.0f)));
}

__m256 Sigmoid(const __m256 a, const __m256 b) {
  return lasx_mul_f32(lasx_mul_f32(a, b),
                       lasx_sub_f32(lasx_set1_f32(1.0f), b));
}

__m256 Tanh(const __m256 a, const __m256 b) {
  return lasx_mul_f32(
      a, lasx_sub_f32(lasx_set1_f32(1.0f), lasx_mul_f32(b, b)));
}

__m256 Identity(const __m256 a, const __m256 b) { return a; }
}  // namespace lasx
}  // namespace backward

}  // namespace detail
}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle

#endif
