// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/loongarch/math/include/group_norm.h"
#include "lite/backends/loongarch/xxl.h"
#include <stdio.h>
#include <cmath>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

void group_norm(const float* in,
                float* out,
                const int n,
                const int c,
                const int height,
                const int width,
                const float epsilon,
                const int groups,
                const float* scale,
                const float* bias,
                float* saved_mean,
                float* saved_variance) {
  int nb = n;
  int spatial_size = height * width;
  int group_size = (c - 1) / groups +
                   1;  // equal to instance_norm if the groups value equals to c

// compute saved_mean and saved_variance
#pragma omp parallel for
  for (int i = 0; i < nb; i++) {
    for (int gid = 0; gid < groups; gid++) {
      float sum_spatial = 0.f;
      float summ_spatial = 0.f;
      const float* in_data =
          in + (i * groups + gid) * group_size * spatial_size;  // input offset
      float* out_data =
          out +
          (i * groups + gid) * group_size * spatial_size;  // output offset
      int number = (group_size > (c - gid * group_size))
                       ? (c - gid * group_size)
                       : group_size;

      // for each group
      for (int nid = 0; nid < number; nid++) {
        const float* in_p = in_data + nid * spatial_size;
        // for each image size w x h
        for (int h = 0; h < height; ++h) {
          int w = width;

          __m128 sum0 = lsx_set1_f32(0.f);
          __m128 sum1 = lsx_set1_f32(0.f);
          __m128 sum2 = lsx_set1_f32(0.f);
          __m128 sum3 = lsx_set1_f32(0.f);
          __m128 square_sum0 = lsx_set1_f32(0.f);
          __m128 square_sum1 = lsx_set1_f32(0.f);
          __m128 square_sum2 = lsx_set1_f32(0.f);
          __m128 square_sum3 = lsx_set1_f32(0.f);
          __m128 in0, in1, in2, in3;
          for (; w > 15; w -= 16) {
            in0 = lsx_loadu_f32(in_p);
            in1 = lsx_loadu_f32(in_p + 4);
            in2 = lsx_loadu_f32(in_p + 8);
            in3 = lsx_loadu_f32(in_p + 12);
            // add x
            sum0 = lsx_add_f32(sum0, in0);
            sum1 = lsx_add_f32(sum1, in1);
            sum2 = lsx_add_f32(sum2, in2);
            sum3 = lsx_add_f32(sum3, in3);
            // add x * x
            square_sum0 = lsx_fmadd_f32(in0, in0, square_sum0);
            square_sum1 = lsx_fmadd_f32(in1, in1, square_sum1);
            square_sum2 = lsx_fmadd_f32(in2, in2, square_sum2);
            square_sum3 = lsx_fmadd_f32(in3, in3, square_sum3);

            in_p += 16;
          }
          for (; w > 7; w -= 8) {
            in0 = lsx_loadu_f32(in_p);
            in1 = lsx_loadu_f32(in_p + 4);
            sum0 = lsx_add_f32(sum0, in0);
            sum1 = lsx_add_f32(sum1, in1);
            square_sum0 = lsx_fmadd_f32(in0, in0, square_sum0);
            square_sum1 = lsx_fmadd_f32(in1, in1, square_sum1);
            in_p += 8;
          }
          for (; w > 3; w -= 4) {
            in0 = lsx_loadu_f32(in_p);
            sum0 = lsx_add_f32(sum0, in0);
            square_sum0 = lsx_fmadd_f32(in0, in0, square_sum0);
            in_p += 4;
          }
          float sum = 0.f;
          float summ = 0.f;
          for (; w > 0; w--) {
            sum += *in_p;
            summ += (*in_p) * (*in_p);
            in_p++;
          }

          sum0 = lsx_add_f32(sum0, sum1);
          sum2 = lsx_add_f32(sum2, sum3);
          square_sum0 = lsx_add_f32(square_sum0, square_sum1);
          square_sum2 = lsx_add_f32(square_sum2, square_sum3);

          sum0 = lsx_add_f32(sum0, sum2);
          square_sum0 = lsx_add_f32(square_sum0, square_sum2);

          __m128 r = lsx_hadd_f32(sum0, square_sum0);
          r = lsx_hadd_f32(r, r);
          float buf[4];
          lsx_storeu_f32(buf, r);
          sum += buf[0];
          summ += buf[1];
          // accumulation
          sum_spatial += sum;
          summ_spatial += summ;
        }
      }

      float mean = sum_spatial / (number * spatial_size);
      // float x_var = summ_spatial / (number * spatial_size);
      // float variance = summ_spatial / (number * spatial_size) - mean * mean;
      // the flolowing code has higher precision than above comment code
      float variance = (summ_spatial - mean * mean * spatial_size * number) /
                       (number * spatial_size);
      float std = 1.f / sqrtf(variance + epsilon);

      saved_mean[i * groups + gid] = mean;
      saved_variance[i * groups + gid] = variance;

      // compute each group_norm result: out = scale * (in - mean) / std + bias
      for (int nid = 0; nid < number; nid++) {
        const float* in_p = in_data + nid * spatial_size;
        float* out_p = out_data + nid * spatial_size;

        int j = spatial_size;
        const float sstd_val =
            scale == nullptr ? std : scale[gid * group_size + nid] * std;
        const float bias_val =
            bias == nullptr ? 0. : bias[gid * group_size + nid];
        const float mean_val = mean;
        const __m128 vsstd = lsx_set1_f32(sstd_val);
        const __m128 vbias = lsx_set1_f32(bias_val);
        const __m128 vmean = lsx_set1_f32(mean_val);
        __m128 in0, in1, submean0, submean1, out0, out1;

        for (; j > 7; j -= 8) {
          in0 = lsx_loadu_f32(in_p);
          in1 = lsx_loadu_f32(in_p + 4);
          submean0 = lsx_sub_f32(in0, vmean);
          submean1 = lsx_sub_f32(in1, vmean);
          out0 = lsx_fmadd_f32(submean0, vsstd, vbias);
          out1 = lsx_fmadd_f32(submean1, vsstd, vbias);

          lsx_storeu_f32(out_p, out0);
          lsx_storeu_f32(out_p + 4, out1);

          in_p += 8;
          out_p += 8;
        }
        for (; j > 3; j -= 4) {
          in0 = lsx_loadu_f32(in_p);
          submean0 = lsx_sub_f32(in0, vmean);
          out0 = lsx_fmadd_f32(submean0, vsstd, vbias);

          lsx_storeu_f32(out_p, out0);

          in_p += 4;
          out_p += 4;
        }
        for (; j > 0; j--) {
          *out_p = (*in_p - mean_val) * sstd_val + bias_val;
          in_p++;
          out_p++;
        }
      }
    }
  }
}
}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
