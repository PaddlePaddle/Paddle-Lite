/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef BATCHNORM_OP

#pragma once

#include <cmath>
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void BatchnormCompute(const BatchNormParam<CPU> &param) {
  const Tensor *input_x = param.InputX();
  auto input_x_ptr = input_x->data<float>();
  const auto &x_dims = input_x->dims();
  const int N = x_dims[0];
  const int C = x_dims[1];
  const int H = x_dims[2];
  const int W = x_dims[3];
  const int stride0 = C * H * W;
  const int stride1 = H * W;
  const int stride2 = W;
  Tensor *out = param.OutputY();
  auto out_ptr = out->mutable_data<float>();
  const float epsilon = param.Epsilon();
  const Tensor *mean = param.InputMean();
  const Tensor *variance = param.InputVariance();
  const Tensor *scale = param.InputScale();
  const Tensor *bias = param.InputBias();
  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  //  Tensor inv_std;
  //  auto inv_std_ptr = inv_std.mutable_data<float>(make_ddim({C}));

  PADDLE_MOBILE_ENFORCE(C == variance->numel(),
                        "C must equal to variance.numel()");

  int HXW = H * W;

#if __ARM_NEON
#if __aarch64__
  float *inv_std_ptr = new float[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }

  Tensor new_scale;
  auto new_scale_ptr = new_scale.mutable_data<float>(framework::make_ddim({C}));
  Tensor new_bias;
  auto new_bias_ptr = new_bias.mutable_data<float>(framework::make_ddim({C}));

  /// ((x - est_mean) * (inv_var) * scale + bias equal to
  /// (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
    {
      for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
          int tmp_index = n * stride0 + i * stride1 + h * stride2;
          for (int w = 0; w < W; w++) {
            int index = tmp_index + w;
            out_ptr[index] =
                input_x_ptr[index] * new_scale_ptr[i] + new_bias_ptr[i];
          }
        }
      }
    }
  }
  delete[] inv_std_ptr;
#else

  if (HXW > 32) {
    int NXC = N * C;
    float *inv_std_ptr = new float[NXC * 4];
    float *volatile new_scale_ptr = new float[NXC * 4];
    float *volatile new_bias_ptr = new float[NXC * 4];

    /// std = (var + epsilon).sqrt();
    /// inv_std = 1 / std;
    for (int i = 0; i < C * 4; i += 4) {
      int index = i / 4;
      inv_std_ptr[i] =
          1 / static_cast<float>(pow((variance_ptr[index] + epsilon), 0.5));
      inv_std_ptr[i + 1] = inv_std_ptr[i];
      inv_std_ptr[i + 2] = inv_std_ptr[i];
      inv_std_ptr[i + 3] = inv_std_ptr[i];

      new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[index];
      new_scale_ptr[i + 1] = new_scale_ptr[i];
      new_scale_ptr[i + 2] = new_scale_ptr[i];
      new_scale_ptr[i + 3] = new_scale_ptr[i];

      new_bias_ptr[i] =
          bias_ptr[index] - mean_ptr[index] * inv_std_ptr[i] * scale_ptr[index];

      new_bias_ptr[i + 1] = new_bias_ptr[i];
      new_bias_ptr[i + 2] = new_bias_ptr[i];
      new_bias_ptr[i + 3] = new_bias_ptr[i];
    }

    for (int j = C * 4; j < NXC * 4; ++j) {
      new_scale_ptr[j] = new_scale_ptr[j - C * 4];
      new_bias_ptr[j] = new_bias_ptr[j - C * 4];
    }

    asm volatile(
        "subs %[N], %[N], #1                  \n\t"
        "blt        end_n_%=                  \n\t"
        "loop_n_%=:                           \n\t"

        "subs %[C], %[C], #1                   \n\t"
        "blt        end_c_%=                  \n\t"
        "loop_c_%=:                           \n\t"

        "vld1.32 {q9}, [%[new_scale_ptr]]!    \n\t"
        "vld1.32 {q10}, [%[new_bias_ptr]]!    \n\t"

        "mov r6, %[HXW]       \n\t"

        "subs r6, r6, #32                       \n\t"
        "blt        end_hw_%=                   \n\t"
        "loop_hw_%=:                            \n\t"

        "vld1.32 {q1, q2}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q3, q4}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q5, q6}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q7, q8}, [%[input_x_ptr]]!    \n\t"

        "vmul.f32   q1, q1,   q9  \n\t"
        "vmul.f32   q2, q2,   q9  \n\t"
        "vmul.f32   q3, q3,   q9  \n\t"
        "vmul.f32   q4, q4,   q9  \n\t"

        "vmul.f32   q5, q5,   q9  \n\t"
        "vmul.f32   q6, q6,   q9  \n\t"
        "vmul.f32   q7, q7,   q9  \n\t"
        "vmul.f32   q8, q8,   q9  \n\t"

        "vadd.f32   q1,  q1,  q10 \n\t"
        "vadd.f32   q2, q2,   q10  \n\t"
        "vadd.f32   q3, q3,   q10  \n\t"
        "vadd.f32   q4,  q4,  q10 \n\t"
        "vadd.f32   q5,  q5,  q10 \n\t"
        "vadd.f32   q6,  q6,  q10 \n\t"
        "vadd.f32   q7,  q7,  q10 \n\t"
        "vadd.f32   q8,  q8,  q10 \n\t"

        "vst1.32 {q1, q2}, [%[out_ptr]]!        \n\t"
        "vst1.32 {q3, q4}, [%[out_ptr]]!       \n\t"
        "vst1.32 {q5, q6}, [%[out_ptr]]!       \n\t"
        "vst1.32 {q7, q8}, [%[out_ptr]]!       \n\t"

        "subs r6, r6, #32                    \n\t"
        "bge        loop_hw_%=                \n\t"
        "end_hw_%=:                           \n\t"

        "cmp  r6, #0                                \n\t"
        "bge  end_remainder_%=                      \n\t"
        "mov r5, #4                             \n\t"
        "mul  r6, r6, r5                            \n\t"
        "add %[input_x_ptr], %[input_x_ptr], r6     \n\t"

        "vld1.32 {q1, q2}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q3, q4}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q5, q6}, [%[input_x_ptr]]!    \n\t"
        "vld1.32 {q7, q8}, [%[input_x_ptr]]!    \n\t"

        "vmul.f32   q1, q1,   q9  \n\t"
        "vmul.f32   q2, q2,   q9  \n\t"
        "vmul.f32   q3, q3,   q9  \n\t"
        "vmul.f32   q4, q4,   q9  \n\t"
        "vmul.f32   q5, q5,   q9  \n\t"
        "vmul.f32   q6, q6,   q9  \n\t"
        "vmul.f32   q7, q7,   q9  \n\t"
        "vmul.f32   q8, q8,   q9  \n\t"
        "vadd.f32   q1,  q1,  q10 \n\t"
        "vadd.f32   q2, q2,   q10  \n\t"
        "vadd.f32   q3, q3,   q10  \n\t"
        "vadd.f32   q4,  q4,  q10 \n\t"
        "vadd.f32   q5,  q5,  q10 \n\t"
        "vadd.f32   q6,  q6,  q10 \n\t"
        "vadd.f32   q7,  q7,  q10 \n\t"
        "vadd.f32   q8,  q8,  q10 \n\t"

        "add %[out_ptr], %[out_ptr], r6         \n\t"
        "vst1.32 {q1, q2}, [%[out_ptr]]!        \n\t"
        "vst1.32 {q3, q4}, [%[out_ptr]]!        \n\t"
        "vst1.32 {q5, q6}, [%[out_ptr]]!        \n\t"
        "vst1.32 {q7, q8}, [%[out_ptr]]!        \n\t"

        "end_remainder_%=:                      \n\t"

        "subs %[C], %[C], #1                    \n\t"
        "bge        loop_c_%=                   \n\t"
        "end_c_%=:                              \n\t"

        "subs %[N], %[N], #1                    \n\t"
        "bge        loop_n_%=                   \n\t"
        "end_n_%=:                              \n\t"
        :
        : [input_x_ptr] "r"(input_x_ptr), [out_ptr] "r"(out_ptr),
          [new_scale_ptr] "r"(new_scale_ptr), [new_bias_ptr] "r"(new_bias_ptr),
          [N] "r"(N), [C] "r"(C), [HXW] "r"(HXW)
        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
          "q10", "r5", "r6");

    delete[] inv_std_ptr;
    delete[] new_scale_ptr;
    delete[] new_bias_ptr;

  } else {
    float *inv_std_ptr = new float[C];
    for (int i = 0; i < C; i++) {
      inv_std_ptr[i] =
          1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
    }

    Tensor new_scale;
    auto new_scale_ptr =
        new_scale.mutable_data<float>(framework::make_ddim({C}));
    Tensor new_bias;
    auto new_bias_ptr = new_bias.mutable_data<float>(framework::make_ddim({C}));

    /// ((x - est_mean) * (inv_var) * scale + bias equal to
    /// (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
    for (int i = 0; i < C; i++) {
      new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
      new_bias_ptr[i] =
          bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
      {
        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            int tmp_index = n * stride0 + i * stride1 + h * stride2;
            for (int w = 0; w < W; w++) {
              int index = tmp_index + w;
              out_ptr[index] =
                  input_x_ptr[index] * new_scale_ptr[i] + new_bias_ptr[i];
            }
          }
        }
      }
    }

    delete[] inv_std_ptr;
  }
#endif
#else
  float *inv_std_ptr = new float[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }

  Tensor new_scale;
  auto new_scale_ptr = new_scale.mutable_data<float>(framework::make_ddim({C}));
  Tensor new_bias;
  auto new_bias_ptr = new_bias.mutable_data<float>(framework::make_ddim({C}));

  /// ((x - est_mean) * (inv_var) * scale + bias equal to
  /// (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
    {
      for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
          int tmp_index = n * stride0 + i * stride1 + h * stride2;
          for (int w = 0; w < W; w++) {
            int index = tmp_index + w;
            out_ptr[index] =
                input_x_ptr[index] * new_scale_ptr[i] + new_bias_ptr[i];
          }
        }
      }
    }
  }
  delete[] inv_std_ptr;
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
