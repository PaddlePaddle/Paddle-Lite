// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/fp16/fill_bias_act_fp16.h"
#include <arm_neon.h>
#include <algorithm>

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
// clang-format off
#ifdef __aarch64__
#define FILL_BIAS_CNT0                                          \
  "cmp %w[cnt_num0], #1             \n"                         \
  "ldr q0, [%[din_ptr]], #16        \n" /*vld1q_f16(din_ptr0)*/ \
  "blt 1f                           \n"                         \
  "0:                               \n"                         \
  "ldr q1, [%[din_ptr]], #16        \n" /*vld1q_f16(din_ptr0)*/ \
  "ldr q2, [%[din_ptr]], #16        \n" /*vld1q_f16(din_ptr0)*/ \
  "ldr q3, [%[din_ptr]], #16        \n" /*vld1q_f16(din_ptr0)*/ \
  "fadd v0.8h, v0.8h, %[vbias].8h   \n"                         \
  "fadd v1.8h, v1.8h, %[vbias].8h   \n"                         \
  "fadd v2.8h, v2.8h, %[vbias].8h   \n"                         \
  "fadd v3.8h, v3.8h, %[vbias].8h   \n"

#define FILL_RELU_CNT0                                    \
  "fmax v0.8h, v0.8h, %[vzero].8h   \n" /* vmaxq_f16() */ \
  "fmax v1.8h, v1.8h, %[vzero].8h   \n" /* vmaxq_f16() */ \
  "fmax v2.8h, v2.8h, %[vzero].8h   \n" /* vmaxq_f16() */ \
  "fmax v3.8h, v3.8h, %[vzero].8h   \n" /* vmaxq_f16() */
#define FILL_RELU6_CNT0                                    \
  "fmin v0.8h, v0.8h, %[vsix].8h   \n" /* vminq_f16() */ \
  "fmin v1.8h, v1.8h, %[vsix].8h   \n" /* vminq_f16() */ \
  "fmin v2.8h, v2.8h, %[vsix].8h   \n" /* vminq_f16() */ \
  "fmin v3.8h, v3.8h, %[vsix].8h   \n" /* vminq_f16() */
#define FILL_LEAKY_RELU_CNT0                             \
  "fcmge v4.8h, v0.8h,  %[vzero].8h  \n" /* vcgeq_f16 */ \
  "fmul v5.8h, v0.8h, %[vscale].8h   \n" /* vmulq_f16 */ \
  "fcmge v6.8h, v1.8h,  %[vzero].8h  \n" /* vcgeq_f16 */ \
  "fmul v7.8h, v1.8h, %[vscale].8h   \n" /* vmulq_f16 */ \
  "fcmge v8.8h, v2.8h,  %[vzero].8h  \n" /* vcgeq_f16 */ \
  "fmul v9.8h, v2.8h, %[vscale].8h   \n" /* vmulq_f16 */ \
  "fcmge v10.8h, v3.8h,  %[vzero].8h \n" /* vcgeq_f16 */ \
  "fmul v11.8h, v3.8h, %[vscale].8h  \n" /* vmulq_f16 */ \
  "bif v0.16b, v5.16b, v4.16b        \n" /* choose*/     \
  "bif v1.16b, v7.16b, v6.16b        \n" /* choose*/     \
  "bif v2.16b, v9.16b, v8.16b        \n" /* choose*/     \
  "bif v3.16b, v11.16b, v10.16b      \n" /* choose*/
#define FILL_STORE_CNT0                                  \
  "subs %w[cnt_num0], %w[cnt_num0], #1\n"                \
  "str q0, [%[dout_ptr]], #16      \n" /* vst1q_f16() */ \
  "str q1, [%[dout_ptr]], #16      \n" /* vst1q_f16() */ \
  "ldr q0, [%[din_ptr]], #16       \n"                   \
  "str q2, [%[dout_ptr]], #16      \n" /* vst1q_f16() */ \
  "str q3, [%[dout_ptr]], #16      \n" /* vst1q_f16() */ \
  "bne  0b                         \n"                   \
  "1:                               \n"                  \
  "cmp %w[cnt_num1], #1             \n"                  \
  "blt 2f                           \n"                  \
  "3:                               \n"                  \
  "fadd v0.8h, v0.8h, %[vbias].8h   \n"
#define  FILL_RELU_CNT1  "fmax v0.8h, v0.8h, %[vzero].8h   \n" /* vmaxq_f16() */
#define  FILL_RELU6_CNT1 "fmin v0.8h, v0.8h, %[vsix].8h   \n" /* vminq_f16() */
#define  FILL_LEAKY_RELU_CNT1                            \
  "fcmge v4.8h, v0.8h,  %[vzero].8h  \n" /* vcgeq_f16 */ \
  "fmul v5.8h, v0.8h, %[vscale].8h   \n" /* vmulq_f16 */ \
  "bif v0.16b, v5.16b, v4.16b        \n" /* choose*/
#define FILL_STORE_CNT1                                  \
  "subs %w[cnt_num1], %w[cnt_num1], #1\n"                \
  "str q0, [%[dout_ptr]], #16       \n" /* vst1q_f16() */\
  "ldr q0, [%[din_ptr]], #16        \n"                  \
  "bne  3b                          \n"                  \
  "2:                               \n"                  \
  "sub %[din_ptr], %[din_ptr], #16  \n"
#else
#endif
// clang-format on

template <>
void fill_bias_act_fp16<float16_t>(
    float16_t* tensor,
    const float16_t* bias,
    int channel,
    int channel_size,
    bool flag_bias,
    const operators::ActivationParam* act_param) {
  float16_t* data = tensor;
  int cnt_32 = channel_size >> 5;
  int rem_32 = channel_size & 31;
  int cnt_8 = rem_32 >> 3;
  int rem_8 = rem_32 & 7;
  float16x8_t vzero = vdupq_n_f16(0.f);
  if (act_param != nullptr && act_param->has_active) {
    float16x8_t vsix = vdupq_n_f16(act_param->Relu_clipped_coef);
    float16x8_t vscale = vdupq_n_f16(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
        for (int j = 0; j < channel; j++) {
          float16_t bias_data = flag_bias ? bias[j] : 0.f;
          float16_t* src = data + j * channel_size;
          float16_t* dst = data + j * channel_size;
          float16x8_t vbias = vdupq_n_f16(bias_data);
          int cnt_num0 = cnt_32;
          int cnt_num1 = cnt_8;
#ifdef __aarch64__
          asm volatile(FILL_BIAS_CNT0 FILL_RELU_CNT0 FILL_STORE_CNT0
                           FILL_RELU_CNT1 FILL_STORE_CNT1
                       : [din_ptr] "+r"(src),
                         [dout_ptr] "+r"(dst),
                         [cnt_num0] "+r"(cnt_num0),
                         [cnt_num1] "+r"(cnt_num1)
                       : [vzero] "w"(vzero), [vbias] "w"(vbias)
                       : "memory", "cc", "v0", "v1", "v2", "v3");
#endif
          for (int i = 0; i < rem_8; i++) {
            float16_t tmp = (*src + bias_data);
            *dst = tmp >= 0.f ? tmp : 0.f;
            src++;
            dst++;
          }
        }
        break;
      case lite_api::ActivationType::kRelu6:
        for (int j = 0; j < channel; j++) {
          float16_t bias_data = flag_bias ? bias[j] : 0.f;
          float16_t* src = data + j * channel_size;
          float16_t* dst = data + j * channel_size;
          float16x8_t vbias = vdupq_n_f16(bias_data);
          int cnt_num0 = cnt_32;
          int cnt_num1 = cnt_8;
#ifdef __aarch64__
          asm volatile(
              FILL_BIAS_CNT0 FILL_RELU_CNT0 FILL_RELU6_CNT0 FILL_STORE_CNT0
                  FILL_RELU_CNT1 FILL_RELU6_CNT1 FILL_STORE_CNT1
              : [din_ptr] "+r"(src),
                [dout_ptr] "+r"(dst),
                [cnt_num0] "+r"(cnt_num0),
                [cnt_num1] "+r"(cnt_num1)
              : [vzero] "w"(vzero), [vbias] "w"(vbias), [vsix] "w"(vsix)
              : "memory", "cc", "v0", "v1", "v2", "v3");
#endif
          for (int i = 0; i < rem_8; i++) {
            float16_t tmp = (*src + bias_data);
            tmp = tmp >= 0.f ? tmp : 0.f;
            *dst = tmp <= act_param->Relu_clipped_coef
                       ? tmp
                       : act_param->Relu_clipped_coef;
            src++;
            dst++;
          }
        }
        break;
      case lite_api::ActivationType::kLeakyRelu:
        for (int j = 0; j < channel; j++) {
          float16_t bias_data = flag_bias ? bias[j] : 0.f;
          float16_t* src = data + j * channel_size;
          float16_t* dst = data + j * channel_size;
          float16x8_t vbias = vdupq_n_f16(bias_data);
          int cnt_num0 = cnt_32;
          int cnt_num1 = cnt_8;
#ifdef __aarch64__
          asm volatile(
              FILL_BIAS_CNT0 FILL_LEAKY_RELU_CNT0 FILL_STORE_CNT0
                  FILL_LEAKY_RELU_CNT1 FILL_STORE_CNT1
              : [din_ptr] "+r"(src),
                [dout_ptr] "+r"(dst),
                [cnt_num0] "+r"(cnt_num0),
                [cnt_num1] "+r"(cnt_num1)
              : [vzero] "w"(vzero), [vbias] "w"(vbias), [vscale] "w"(vscale)
              : "memory",
                "cc",
                "v0",
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v6",
                "v7",
                "v8",
                "v9",
                "v10",
                "v11");
#endif
          for (int i = 0; i < rem_8; i++) {
            float16_t tmp = (*src + bias_data);
            if (tmp >= 0.f) {
              *dst = tmp;
            } else {
              *dst = tmp * act_param->Leaky_relu_alpha;
            }
            src++;
            dst++;
          }
        }
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param->active_type)
                   << " fuse not support";
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      float16_t bias_data = flag_bias ? bias[j] : 0.f;
      float16_t* src = data + j * channel_size;
      float16_t* dst = data + j * channel_size;
      float16x8_t vbias = vdupq_n_f16(bias_data);
      int cnt_num0 = cnt_32;
      int cnt_num1 = cnt_8;
#ifdef __aarch64__
      asm volatile(FILL_BIAS_CNT0 FILL_STORE_CNT0 FILL_STORE_CNT1
                   : [din_ptr] "+r"(src),
                     [dout_ptr] "+r"(dst),
                     [cnt_num0] "+r"(cnt_num0),
                     [cnt_num1] "+r"(cnt_num1)
                   : [vbias] "w"(vbias)
                   : "memory", "cc", "v0", "v1", "v2", "v3");
#endif
      for (int i = 0; i < rem_8; i++) {
        *dst = *src + bias_data;
        dst++;
        src++;
      }
    }
  }
}
#ifdef __aarch64__
#undef FILL_BIAS_CNT0
#undef FILL_RELU_CNT0
#undef FILL_RELU6_CNT0
#undef FILL_LEAKY_RELU_CNT0
#undef FILL_STORE_CNT0
#undef FILL_RELU_CNT1
#undef FILL_RELU6_CNT1
#undef FILL_LEAKY_RELU_CNT1
#undef FILL_STORE_CNT1
#endif
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
