/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "lite/backends/loongarch/jit/more/intrinsic/layer_norm.h"
#include <limits>
#include "lite/backends/loongarch/jit/registry.h"

namespace paddle {
namespace lite {
namespace jit {
namespace more {
namespace intrinsic {

void LayerNorm(float* x,
               float* out,
               float* mean,
               float* var,
               const float* scale,
               const float* bias,
               int height,
               const float epsilon,
               int right) {
  __m256 sum;
  __m256 mean_vec, var_vec;
  __m128 hi, lo;
  __m256 tmp;
  size_t offset;
  size_t j;
  int block = LASX_FLOAT_BLOCK;
  const int rest = right % block;
  const int end = right - rest;

  __m256 reverse_num_vec =
      lasx_div_f32(lasx_set1_f32(1.0), lasx_set1_f32(right));
  __m256 epsilon_vec = lasx_set1_f32(epsilon);
  int rest_mask =
      ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
  __m256i mask_vec = lasx_set_i32(rest_mask & 0x80 ? 0xffffffff : 0,
                                      rest_mask & 0x40 ? 0xffffffff : 0,
                                      rest_mask & 0x20 ? 0xffffffff : 0,
                                      rest_mask & 0x10 ? 0xffffffff : 0,
                                      rest_mask & 0x8 ? 0xffffffff : 0,
                                      rest_mask & 0x4 ? 0xffffffff : 0,
                                      rest_mask & 0x2 ? 0xffffffff : 0,
                                      rest_mask & 0x1 ? 0xffffffff : 0);

  for (int i = 0; i < height; ++i) {
    offset = i * right;

    /* get mean */
    sum = lasx_setzero_f32();
    for (j = offset; j < end + offset; j += block) {
      sum = lasx_add_f32(sum, lasx_loadu_f32((const float*)x + j));
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = lasx_loadu_f32((const float*)x + j);
      tmp = lasx_blendv_f32(lasx_setzero_f32(),
                             tmp,
                             *(__m256*)&mask_vec);  // NOLINT
      sum = lasx_add_f32(sum, tmp);
    }
    hi = lasx_extractf128_f32(sum, 1);
    lo = lasx_extractf128_f32(sum, 0);
    sum = lasx_add_f32(
        sum,
        lasx_insertf128_f32(
            lasx_insertf128_f32(lasx_setzero_f32(), hi, 0), lo, 1));
    sum = lasx_hadd_f32(sum, sum);
    sum = lasx_hadd_f32(sum, sum);
    mean_vec = lasx_mul_f32(sum, reverse_num_vec);
    mean[i] = *reinterpret_cast<float*>(&mean_vec);

    /* get variance */
    sum = lasx_setzero_f32();
    for (j = offset; j < end + offset; j += block) {
      tmp = lasx_sub_f32(lasx_loadu_f32((const float*)x + j), mean_vec);
      tmp = lasx_mul_f32(tmp, tmp);
      sum = lasx_add_f32(sum, tmp);
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = lasx_sub_f32(lasx_loadu_f32((const float*)x + j), mean_vec);
      tmp = lasx_mul_f32(tmp, tmp);
      tmp = lasx_blendv_f32(lasx_setzero_f32(),
                             tmp,
                             *(__m256*)&mask_vec);  // NOLINT
      sum = lasx_add_f32(sum, tmp);
    }
    hi = lasx_extractf128_f32(sum, 1);
    lo = lasx_extractf128_f32(sum, 0);
    sum = lasx_add_f32(
        sum,
        lasx_insertf128_f32(
            lasx_insertf128_f32(lasx_setzero_f32(), hi, 0), lo, 1));
    sum = lasx_hadd_f32(sum, sum);
    sum = lasx_hadd_f32(sum, sum);
    var_vec = lasx_mul_f32(sum, reverse_num_vec);
    var[i] = *reinterpret_cast<float*>(&var_vec);

    /* get x_norm and calculate output*/
    for (j = offset; j < end + offset; j += block) {
      tmp = lasx_sub_f32(lasx_loadu_f32((const float*)x + j), mean_vec);
      tmp = lasx_div_f32(tmp,
                          lasx_sqrt_f32(lasx_add_f32(var_vec, epsilon_vec)));
      lasx_storeu_f32(reinterpret_cast<float*>(out) + j, tmp);
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = lasx_sub_f32(lasx_loadu_f32((const float*)x + j), mean_vec);
      tmp = lasx_div_f32(tmp,
                          lasx_sqrt_f32(lasx_add_f32(var_vec, epsilon_vec)));
      lasx_storeu_f32(reinterpret_cast<float*>(out) + j, tmp);
    }

    if (scale) {
      if (rest != 0) {
        j = offset + right - block;
        tmp = lasx_loadu_f32((const float*)out + j);
      }
      for (j = offset; j < end + offset; j += block) {
        lasx_storeu_f32(
            reinterpret_cast<float*>(out) + j,
            lasx_mul_f32(lasx_loadu_f32((const float*)out + j),
                          lasx_loadu_f32((const float*)scale + j - offset)));
      }
      if (rest != 0) {
        j = offset + right - block;
        lasx_storeu_f32(
            reinterpret_cast<float*>(out) + j,
            lasx_mul_f32(tmp,
                          lasx_loadu_f32((const float*)scale + j - offset)));
      }
    }

    if (bias) {
      if (rest != 0) {
        j = offset + right - block;
        tmp = lasx_loadu_f32((const float*)out + j);
      }
      for (j = offset; j < end + offset; j += block) {
        lasx_storeu_f32(
            reinterpret_cast<float*>(out) + j,
            lasx_add_f32(lasx_loadu_f32((const float*)out + j),
                          lasx_loadu_f32((const float*)bias + j - offset)));
      }
      if (rest != 0) {
        j = offset + right - block;
        lasx_storeu_f32(
            reinterpret_cast<float*>(out) + j,
            lasx_add_f32(tmp,
                          lasx_loadu_f32((const float*)bias + j - offset)));
      }
    }
  }
}

bool LayerNormKernel::CanBeUsed(const int& d) const {
  return loongarch::MayIUse(loongarch::lasx) && d >= LASX_FLOAT_BLOCK;
}

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace lite
}  // namespace paddle

namespace intrinsic = paddle::lite::jit::more::intrinsic;

REGISTER_JITKERNEL_MORE(kLayerNorm, intrinsic, intrinsic::LayerNormKernel);
