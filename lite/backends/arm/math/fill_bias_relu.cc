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

#include "lite/backends/arm/math/fill_bias_relu.h"
#include <algorithm>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

// clang-format off
#ifdef __aarch64__
#define FILL_BIAS                                               \
  "1:                               \n"                         \
  "ld1 {v0.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "ld1 {v1.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "ld1 {v2.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "ld1 {v3.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "fadd v0.4s, v0.4s, %[vbias].4s   \n"                         \
  "fadd v1.4s, v1.4s, %[vbias].4s   \n"                         \
  "fadd v2.4s, v2.4s, %[vbias].4s   \n"                         \
  "fadd v3.4s, v3.4s, %[vbias].4s   \n"

#define INT32_TO_FP32                                                       \
  "1:                               \n"                                     \
  "ld1 {v0.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/             \
  "ld1 {v1.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/             \
  "ld1 {v2.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/             \
  "ld1 {v3.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/             \
  /* int32 -> fp32 */                                                       \
  "scvtf   v4.4s, v0.4s\n"                                                  \
  "scvtf   v5.4s, v1.4s\n"                                                  \
  "dup    v0.4s, %[vbias].s[0]\n"                                           \
  "dup    v1.4s, %[vbias].s[0]\n"                                           \
  "scvtf   v6.4s, v2.4s\n"                                                  \
  "scvtf   v7.4s, v3.4s\n"                                                  \
  "dup    v2.4s, %[vbias].s[0]\n"                                           \
  "dup    v3.4s, %[vbias].s[0]\n"                                           \
  /* mul scale */                                                           \
  "fmla v0.4s, v4.4s, %[vscale_val].4s\n"                                   \
  "fmla v1.4s, v5.4s, %[vscale_val].4s\n"                                   \
  "fmla v2.4s, v6.4s, %[vscale_val].4s\n"                                   \
  "fmla v3.4s, v7.4s, %[vscale_val].4s\n"

#define FILL_RELU                                         \
  "fmax v0.4s, v0.4s, %[vzero].4s   \n" /* vmaxq_f32() */ \
  "fmax v1.4s, v1.4s, %[vzero].4s   \n" /* vmaxq_f32() */ \
  "fmax v2.4s, v2.4s, %[vzero].4s   \n" /* vmaxq_f32() */ \
  "fmax v3.4s, v3.4s, %[vzero].4s   \n" /* vmaxq_f32() */
#define FILL_RELU6                                       \
  "fmin v0.4s, v0.4s, %[vsix].4s   \n" /* vmaxq_f32() */ \
  "fmin v1.4s, v1.4s, %[vsix].4s   \n" /* vmaxq_f32() */ \
  "fmin v2.4s, v2.4s, %[vsix].4s   \n" /* vmaxq_f32() */ \
  "fmin v3.4s, v3.4s, %[vsix].4s   \n" /* vmaxq_f32() */
#define FILL_LEAKY_RELU                                  \
  "fcmge v4.4s, v0.4s,  %[vzero].4s  \n" /* vcgeq_f32 */ \
  "fmul v5.4s, v0.4s, %[vscale].4s   \n" /* vmulq_f32 */ \
  "fcmge v6.4s, v1.4s,  %[vzero].4s  \n" /* vcgeq_f32 */ \
  "fmul v7.4s, v1.4s, %[vscale].4s   \n" /* vmulq_f32 */ \
  "fcmge v8.4s, v2.4s,  %[vzero].4s  \n" /* vcgeq_f32 */ \
  "fmul v9.4s, v2.4s, %[vscale].4s   \n" /* vmulq_f32 */ \
  "fcmge v10.4s, v3.4s,  %[vzero].4s \n" /* vcgeq_f32 */ \
  "fmul v11.4s, v3.4s, %[vscale].4s  \n" /* vmulq_f32 */ \
  "bif v0.16b, v5.16b, v4.16b        \n" /* choose*/     \
  "bif v1.16b, v7.16b, v6.16b        \n" /* choose*/     \
  "bif v2.16b, v9.16b, v8.16b        \n" /* choose*/     \
  "bif v3.16b, v11.16b, v10.16b      \n" /* choose*/

#define FILL_HARD_SWISH                                  \
  "fadd  v8.4s,  v0.4s, %[voffset].4s\n"                 \
  "fadd  v9.4s,  v1.4s, %[voffset].4s\n"                 \
  "fadd  v10.4s, v2.4s, %[voffset].4s\n"                 \
  "fadd  v11.4s, v3.4s, %[voffset].4s\n"                 \
  "fmul  v4.4s,  v0.4s, %[vscale].4s \n"                 \
  "fmul  v5.4s,  v1.4s, %[vscale].4s \n"                 \
  "fmax  v8.4s,  v8.4s, %[vzero].4s\n"                   \
  "fmax  v9.4s,  v9.4s, %[vzero].4s\n"                   \
  "fmul  v6.4s,  v2.4s, %[vscale].4s \n"                 \
  "fmax  v10.4s, v10.4s, %[vzero].4s\n"                  \
  "fmax  v11.4s, v11.4s, %[vzero].4s\n"                  \
  "fmul  v7.4s,  v3.4s, %[vscale].4s \n"                 \
  "fmin  v8.4s,  v8.4s, %[vthreshold].4s\n"              \
  "fmin  v9.4s,  v9.4s, %[vthreshold].4s\n"              \
  "fmin  v10.4s, v10.4s, %[vthreshold].4s\n"             \
  "fmin  v11.4s, v11.4s, %[vthreshold].4s\n"             \
  "fmul  v0.4s,  v4.4s, v8.4s\n"                         \
  "fmul  v1.4s,  v5.4s, v9.4s\n"                         \
  "fmul  v2.4s,  v6.4s, v10.4s\n"                        \
  "fmul  v3.4s,  v7.4s, v11.4s\n"

#define FILL_STORE                                       \
  "subs %w[cnt], %w[cnt], #1                    \n"      \
  "st1 {v0.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v1.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v2.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v3.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "bne  1b                                    \n"
#define FP32_MAX                                       \
  /* data >= -127 */                                   \
  "fcmge v4.4s, v0.4s, %[vmax].4s\n"                   \
  "fcmge v5.4s, v1.4s, %[vmax].4s\n"                   \
  "fcmge v6.4s, v2.4s, %[vmax].4s\n"                   \
  "fcmge v7.4s, v3.4s, %[vmax].4s\n"                   \
  /* choose data */                                    \
  "bif v0.16b, %[vmax].16b, v4.16b\n"                  \
  "bif v1.16b, %[vmax].16b, v5.16b\n"                  \
  "bif v2.16b, %[vmax].16b, v6.16b\n"                  \
  "bif v3.16b, %[vmax].16b, v7.16b\n"
#define FP32_TO_INT8                           \
  /* fp32 - int32 */                           \
  "fcvtas  v4.4s, v0.4s\n"                     \
  "fcvtas  v5.4s, v1.4s\n"                     \
  "fcvtas  v6.4s, v2.4s\n"                     \
  "fcvtas  v7.4s, v3.4s\n"                     \
  /* int32 - int16 */                          \
  "sqxtn   v0.4h, v4.4s\n"                     \
  "sqxtn   v1.4h, v5.4s\n"                     \
  "sqxtn   v2.4h, v6.4s\n"                     \
  "sqxtn   v3.4h, v7.4s\n"                     \
  /* int16 - int8 */                           \
  "sqxtn  v4.8b, v0.8h\n"                      \
  "sqxtn  v5.8b, v1.8h\n"                      \
  "sqxtn  v6.8b, v2.8h\n"                      \
  "sqxtn  v7.8b, v3.8h\n"                      \
  "subs %w[cnt], %w[cnt], #1\n"                \
  /* store */                                  \
  "str    s4, [%[dout_ptr]], #4\n"             \
  "str    s5, [%[dout_ptr]], #4\n"             \
  "str    s6, [%[dout_ptr]], #4\n"             \
  "str    s7, [%[dout_ptr]], #4\n"             \
  "bne  1b\n"
#else
#define FILL_BIAS                                            \
  "1:                               \n"                      \
  "vld1.32 {d6-d7}, [%[din_ptr]]!   @ vld1q_f32(din_ptr) \n" \
  "vld1.32 {d8-d9}, [%[din_ptr]]!   @ vld1q_f32(din_ptr) \n" \
  "vld1.32 {d10-d11}, [%[din_ptr]]! @ vld1q_f32(din_ptr) \n" \
  "vld1.32 {d12-d13}, [%[din_ptr]]! @ vld1q_f32(din_ptr) \n" \
  "vadd.f32 q3, q3, %q[vbias] @ add \n"                      \
  "vadd.f32 q4, q4, %q[vbias] @ add \n"                      \
  "vadd.f32 q5, q5, %q[vbias] @ add \n"                      \
  "vadd.f32 q6, q6, %q[vbias] @ add \n"

#define INT32_TO_FP32                                                         \
  "1:                               \n"                                       \
  "vld1.32 {d6-d7}, [%[din_ptr]]!   @ vld1q_f32(din_ptr) \n"                  \
  "vld1.32 {d8-d9}, [%[din_ptr]]!   @ vld1q_f32(din_ptr) \n"                  \
  "vld1.32 {d10-d11}, [%[din_ptr]]! @ vld1q_f32(din_ptr) \n"                  \
  "vld1.32 {d12-d13}, [%[din_ptr]]! @ vld1q_f32(din_ptr) \n"                  \
  /* int32 -> fp32 */                                                         \
  "vcvt.f32.s32 q7, q3 \n"                                                    \
  "vcvt.f32.s32 q8, q4 \n"                                                    \
  "vand.32 q3, %q[vbias], %q[vbias]\n"                                        \
  "vand.32 q4, %q[vbias], %q[vbias]\n"                                        \
  "vcvt.f32.s32 q9, q5 \n"                                                    \
  "vcvt.f32.s32 q10, q6 \n"                                                   \
  "vand.32 q5, %q[vbias], %q[vbias]\n"                                        \
  "vand.32 q6, %q[vbias], %q[vbias]\n"                                        \
  /* mul scale */                                                             \
  "vmla.f32  q3, q7, %q[vscale_val]\n"                                        \
  "vmla.f32  q4, q8, %q[vscale_val]\n"                                        \
  "vmla.f32  q5, q9, %q[vscale_val]\n"                                        \
  "vmla.f32  q6, q10, %q[vscale_val]\n"

#define FILL_RELU                               \
  "vmax.f32 q3, q3, %q[vzero] @ vmaxq_f32() \n" \
  "vmax.f32 q4, q4, %q[vzero] @ vmaxq_f32() \n" \
  "vmax.f32 q5, q5, %q[vzero] @ vmaxq_f32() \n" \
  "vmax.f32 q6, q6, %q[vzero] @ vmaxq_f32() \n"
#define FILL_RELU6                             \
  "vmin.f32 q3, q3, %q[vsix] @ vminq_f32() \n" \
  "vmin.f32 q4, q4, %q[vsix] @ vmaxq_f32() \n" \
  "vmin.f32 q5, q5, %q[vsix] @ vmaxq_f32() \n" \
  "vmin.f32 q6, q6, %q[vsix] @ vmaxq_f32() \n"
#define FILL_LEAKY_RELU                          \
  "vcge.f32 q7, q3, %q[vzero]   @ vcgeq_u32 \n"  \
  "vmul.f32 q8, q3, %q[vscale]  @ vmulq_f32 \n"  \
  "vcge.f32 q9, q4, %q[vzero]   @ vcgeq_u32 \n"  \
  "vmul.f32 q10, q4, %q[vscale]  @ vmulq_f32 \n" \
  "vcge.f32 q11, q5, %q[vzero]   @ vcgeq_u32 \n" \
  "vmul.f32 q12, q5, %q[vscale]  @ vmulq_f32 \n" \
  "vbif q3, q8, q7               @ choose \n"    \
  "vcge.f32 q13, q6, %q[vzero]   @ vcgeq_u32 \n" \
  "vmul.f32 q7, q6, %q[vscale]  @ vmulq_f32 \n" \
  "vbif q4, q10, q9              @ choose \n"    \
  "vbif q5, q12, q11             @ choose \n"    \
  "vbif q6, q7, q13             @ choose \n"

#define FILL_HARD_SWISH                          \
  "vadd.f32  q7,  q3,  %q[offset] @ add \n"      \
  "vadd.f32  q8,  q4,  %q[offset] @ add \n"      \
  "vadd.f32  q9,  q5,  %q[offset] @ add \n"      \
  "vadd.f32  q10, q6,  %q[offset] @ add \n"      \
  "vmul.f32  q11, q3,  %q[scale] \n"             \
  "vmul.f32  q12, q4,  %q[scale] \n"             \
  "vmax.f32  q7,  q7,  %q[vzero] \n"             \
  "vmax.f32  q8,  q8,  %q[vzero] \n"             \
  "vmul.f32  q13, q5,  %q[scale] \n"             \
  "vmax.f32  q9,  q9,  %q[vzero] \n"             \
  "vmax.f32  q10, q10, %q[vzero] \n"             \
  "vmin.f32  q7,  q7,  %q[threshold] \n"         \
  "vmin.f32  q8,  q8,  %q[threshold] \n"         \
  "vmin.f32  q9,  q9,  %q[threshold] \n"         \
  "vmin.f32  q10, q10, %q[threshold] \n"         \
  "vmul.f32  q3,  q7,  q11 \n"                   \
  "vmul.f32  q11, q6,  %q[scale] \n"             \
  "vmul.f32  q4,  q8,  q12 \n"                   \
  "vmul.f32  q5,  q9,  q13 \n"                   \
  "vmul.f32  q6,  q10, q11 \n"
#define FILL_STORE                                          \
  "subs %[cnt], #1                                \n"       \
  "vst1.32 {d6-d7}, [%[dout_ptr]]!       @ vst1q_f32()  \n" \
  "vst1.32 {d8-d9}, [%[dout_ptr]]!       @ vst1q_f32()  \n" \
  "vst1.32 {d10-d11}, [%[dout_ptr]]!     @ vst1q_f32()  \n" \
  "vst1.32 {d12-d13}, [%[dout_ptr]]!     @ vst1q_f32()  \n" \
  "bne  1b                                    \n"
#define FP32_ROUND                  \
  /* roundf */                      \
  "vmov.f32 q11, #-0.5\n"           \
  "vmov.f32 q12, #0.5\n"            \
  "vmov.f32 q13, #0.5\n"            \
  "vcgt.f32   q7, q3, %q[vzero]\n"  \
  "vcgt.f32   q8, q4, %q[vzero]\n"  \
  "vcgt.f32   q9, q5, %q[vzero]\n"  \
  "vcgt.f32   q10, q6, %q[vzero]\n" \
  "vbif.f32   q12, q11, q7\n"       \
  "vbif.f32   q13, q11, q8\n"       \
  "vmov.f32 q7, #0.5\n"             \
  "vmov.f32 q8, #0.5\n"             \
  "vadd.f32   q3, q3, q12\n"        \
  "vadd.f32   q4, q4, q13\n"        \
  "vbif.f32   q7, q11, q9\n"        \
  "vbif.f32   q8, q11, q10\n"       \
  "vadd.f32   q5, q5, q7\n"         \
  "vadd.f32   q6, q6, q8\n"
#define FP32_MAX                 \
  /* data >= -127 */             \
  "vcge.f32 q9, q3, %q[vmax]\n"  \
  "vcge.f32 q10, q4, %q[vmax]\n" \
  "vcge.f32 q11, q5, %q[vmax]\n" \
  "vcge.f32 q12, q6, %q[vmax]\n" \
  "vbif q3, %q[vmax], q9\n"      \
  "vbif q4, %q[vmax], q10\n"     \
  "vbif q5, %q[vmax], q11\n"     \
  "vbif q6, %q[vmax], q12\n"
#define FP32_TO_INT8                             \
  /* fp32 to int32 */                            \
  "vcvt.s32.f32  q7, q3\n"                       \
  "vcvt.s32.f32  q8, q4\n"                       \
  "vcvt.s32.f32  q9, q5\n"                       \
  "vcvt.s32.f32  q10, q6\n"                      \
  /* int32 to int16 */                           \
  "vqmovn.s32 d6, q7\n"                          \
  "vqmovn.s32 d8, q8\n"                          \
  "vqmovn.s32 d10, q9\n"                         \
  "vqmovn.s32 d12, q10\n"                        \
  /* int16 to int8 */                            \
  "vqmovn.s16 d14, q3\n"                         \
  "vqmovn.s16 d16, q4\n"                         \
  "vqmovn.s16 d18, q5\n"                         \
  "vqmovn.s16 d20, q6\n"                         \
  /* store */                                    \
  "subs %[cnt], #1\n"                            \
  "vst1.32    {d14[0]}, [%[dout_ptr]]!\n"        \
  "vst1.32    {d16[0]}, [%[dout_ptr]]!\n"        \
  "vst1.32    {d18[0]}, [%[dout_ptr]]!\n"        \
  "vst1.32    {d20[0]}, [%[dout_ptr]]!\n"        \
  "bne  1b\n"
#endif
// clang-format on

template <>
void fill_bias_act<float>(float* tensor,
                          const float* bias,
                          int channel,
                          int channel_size,
                          bool flag_bias,
                          const operators::ActivationParam* act_param) {
  float* data = tensor;
  int cnt_num = channel_size >> 4;
  int remain = channel_size % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
  if (act_param != nullptr && act_param->has_active) {
    if (act_param->active_type == lite_api::ActivationType::kRelu) {
      for (int j = 0; j < channel; j++) {
        float bias_data = flag_bias ? bias[j] : 0.f;
        float* src = data + j * channel_size;
        float* dst = data + j * channel_size;
        float32x4_t vbias = vdupq_n_f32(bias_data);
        int cnt = cnt_num;
        if (cnt_num > 0) {
#ifdef __aarch64__
          asm volatile(
              FILL_BIAS FILL_RELU FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vbias] "w"(vbias)
              : "memory", "cc", "v0", "v1", "v2", "v3");
#else
          asm volatile(
              FILL_BIAS FILL_RELU FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vbias] "w"(vbias)
              : "memory", "cc", "q3", "q4", "q5", "q6");
#endif
        }
        for (int i = 0; i < remain; i++) {
          float tmp = (*src + bias_data);
          *dst = tmp >= 0.f ? tmp : 0.f;
          src++;
          dst++;
        }
      }
    } else if (act_param->active_type == lite_api::ActivationType::kRelu6) {
      float32x4_t vsix = vdupq_n_f32(act_param->Relu_clipped_coef);
      for (int j = 0; j < channel; j++) {
        float bias_data = flag_bias ? bias[j] : 0.f;
        float* src = data + j * channel_size;
        float* dst = data + j * channel_size;
        float32x4_t vbias = vdupq_n_f32(bias_data);
        int cnt = cnt_num;
        if (cnt_num > 0) {
#ifdef __aarch64__
          asm volatile(
              FILL_BIAS FILL_RELU FILL_RELU6 FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vsix] "w"(vsix), [vbias] "w"(vbias)
              : "memory", "cc", "v0", "v1", "v2", "v3");
#else
          asm volatile(
              FILL_BIAS FILL_RELU FILL_RELU6 FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vsix] "w"(vsix), [vbias] "w"(vbias)
              : "memory", "cc", "q3", "q4", "q5", "q6");
#endif
        }
        for (int i = 0; i < remain; i++) {
          float tmp = (*src + bias_data);
          tmp = tmp >= 0.f ? tmp : 0.f;
          *dst = tmp <= act_param->Relu_clipped_coef
                     ? tmp
                     : act_param->Relu_clipped_coef;
          src++;
          dst++;
        }
      }
    } else if (act_param->active_type == lite_api::ActivationType::kLeakyRelu) {
      float32x4_t vscale = vdupq_n_f32(act_param->Leaky_relu_alpha);
      for (int j = 0; j < channel; j++) {
        float bias_data = flag_bias ? bias[j] : 0.f;
        float* src = data + j * channel_size;
        float* dst = data + j * channel_size;
        float32x4_t vbias = vdupq_n_f32(bias_data);
        int cnt = cnt_num;
        if (cnt_num > 0) {
#ifdef __aarch64__
          asm volatile(
              FILL_BIAS FILL_LEAKY_RELU FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vscale] "w"(vscale), [vbias] "w"(vbias)
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
#else
          asm volatile(
              FILL_BIAS FILL_LEAKY_RELU FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vscale] "w"(vscale), [vbias] "w"(vbias)
              : "memory",
                "cc",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13",
                "q14");
#endif
        }
        for (int i = 0; i < remain; i++) {
          float tmp = (*src + bias_data);
          if (tmp >= 0.f) {
            *dst = tmp;
          } else {
            *dst = tmp * act_param->Leaky_relu_alpha;
          }
          src++;
          dst++;
        }
      }
    } else if (act_param->active_type == lite_api::ActivationType::kHardSwish) {
      float32x4_t vscale =
          div_ps(vdupq_n_f32(1.0f), vdupq_n_f32(act_param->hard_swish_scale));
      float32x4_t voffset = vdupq_n_f32(act_param->hard_swish_offset);
      float32x4_t vthreshold = vdupq_n_f32(act_param->hard_swish_threshold);
      for (int j = 0; j < channel; j++) {
        float bias_data = flag_bias ? bias[j] : 0.f;
        float* src = data + j * channel_size;
        float* dst = data + j * channel_size;
        float32x4_t vbias = vdupq_n_f32(bias_data);
        int cnt = cnt_num;
        if (cnt_num > 0) {
#ifdef __aarch64__
          asm volatile(
              FILL_BIAS FILL_LEAKY_RELU FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero),
                [vscale] "w"(vscale),
                [vbias] "w"(vbias),
                [voffset] "w"(voffset),
                [vthreshold] "w"(vthreshold)
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
#else
          asm volatile(
              FILL_BIAS FILL_LEAKY_RELU FILL_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero),
                [vscale] "w"(vscale),
                [vbias] "w"(vbias),
                [voffset] "w"(voffset),
                [vthreshold] "w"(vthreshold)
              : "memory",
                "cc",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13");
#endif
        }
        for (int i = 0; i < remain; i++) {
          float tmp = (*src + bias_data);
          *dst = std::min(std::max(0.f, tmp + act_param->hard_swish_offset),
                          act_param->hard_swish_threshold) *
                 tmp / act_param->hard_swish_scale;
          src++;
          dst++;
        }
      }
    } else {
      LOG(FATAL) << "this act_type: "
                 << static_cast<int>(act_param->active_type)
                 << " fuse not support";
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vbias = vdupq_n_f32(bias_data);
      float* src = data + j * channel_size;
      float* dst = data + j * channel_size;
      int cnt = cnt_num;
      if (cnt > 0) {
#ifdef __aarch64__
        asm volatile(FILL_BIAS FILL_STORE
                     :
                     [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                     : [vbias] "w"(vbias)
                     : "memory", "cc", "v0", "v1", "v2", "v3");
#else
        asm volatile(FILL_BIAS FILL_STORE
                     :
                     [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                     : [vbias] "w"(vbias)
                     : "memory", "cc", "q3", "q4", "q5", "q6");
#endif
      }
      for (int i = 0; i < remain; i++) {
        *dst = *src + bias_data;
        dst++;
        src++;
      }
    }
  }
}

template <>
void fill_bias_act_calib<float>(float* dout,
                                const int32_t* din,
                                const float* bias,
                                const float* scale,
                                int channel,
                                int channel_size,
                                bool flag_bias,
                                const operators::ActivationParam* act_param) {
  int cnt_num = channel_size >> 4;
  int remain = channel_size & 15;
  float32x4_t vzero = vdupq_n_f32(0.f);
  if (act_param != nullptr && act_param->has_active) {
    float32x4_t vsix = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t vscale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
        for (int j = 0; j < channel; j++) {
          const float bias_data = flag_bias ? bias[j] : 0.f;
          const int32_t* src = din + j * channel_size;
          float* dst = dout + j * channel_size;
          float32x4_t vscale_val = vdupq_n_f32(scale[j]);
          float32x4_t vbias = vdupq_n_f32(bias_data);
          int cnt = cnt_num;
          if (cnt_num > 0) {
#ifdef __aarch64__
            asm volatile(
                INT32_TO_FP32 FILL_RELU FILL_STORE
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "v0",
                  "v1",
                  "v2",
                  "v3",
                  "v4",
                  "v5",
                  "v6",
                  "v7");
#else
            asm volatile(
                INT32_TO_FP32 FILL_RELU FILL_STORE
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10");
#endif
          }
          for (int i = 0; i < remain; i++) {
            float tmp = (*src * scale[j] + bias_data);
            *dst = tmp >= 0.f ? tmp : 0.f;
            src++;
            dst++;
          }
        }
        break;
      case lite_api::ActivationType::kRelu6:
        for (int j = 0; j < channel; j++) {
          const float bias_data = flag_bias ? bias[j] : 0.f;
          const int32_t* src = din + j * channel_size;
          float* dst = dout + j * channel_size;
          float32x4_t vscale_val = vdupq_n_f32(scale[j]);
          float32x4_t vbias = vdupq_n_f32(bias_data);
          int cnt = cnt_num;
          if (cnt_num > 0) {
#ifdef __aarch64__
            asm volatile(
                INT32_TO_FP32 FILL_RELU FILL_RELU6 FILL_STORE
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vsix] "w"(vsix),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "v0",
                  "v1",
                  "v2",
                  "v3",
                  "v4",
                  "v5",
                  "v6",
                  "v7");
#else
            asm volatile(
                INT32_TO_FP32 FILL_RELU FILL_RELU6 FILL_STORE
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vsix] "w"(vsix),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10");
#endif
          }
          for (int i = 0; i < remain; i++) {
            float tmp = (*src * scale[j] + bias_data);
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
          const float bias_data = flag_bias ? bias[j] : 0.f;
          const int32_t* src = din + j * channel_size;
          float* dst = dout + j * channel_size;
          float32x4_t vscale_val = vdupq_n_f32(scale[j]);
          float32x4_t vbias = vdupq_n_f32(bias_data);
          int cnt = cnt_num;
          if (cnt_num > 0) {
#ifdef __aarch64__
            asm volatile(
                INT32_TO_FP32 FILL_LEAKY_RELU FILL_STORE
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale] "w"(vscale),
                  [vscale_val] "w"(vscale_val)
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
#else
            asm volatile(
                INT32_TO_FP32 FILL_LEAKY_RELU FILL_STORE
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale] "w"(vscale),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10",
                  "q11",
                  "q12",
                  "q13",
                  "q14");
#endif
          }
          for (int i = 0; i < remain; i++) {
            float tmp = (*src * scale[j] + bias_data);
            *dst = tmp >= 0.f ? tmp : tmp * act_param->Leaky_relu_alpha;
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
      const float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vscale_val = vdupq_n_f32(scale[j]);
      float32x4_t vbias = vdupq_n_f32(bias_data);
      const int32_t* src = din + j * channel_size;
      float* dst = dout + j * channel_size;
      int cnt = cnt_num;
      if (cnt > 0) {
#ifdef __aarch64__
        asm volatile(
            INT32_TO_FP32 FILL_STORE
            : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
            : [vbias] "w"(vbias), [vscale_val] "w"(vscale_val)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
        asm volatile(
            INT32_TO_FP32 FILL_STORE
            : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
            : [vbias] "w"(vbias), [vscale_val] "w"(vscale_val)
            : "memory", "cc", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10");
#endif
      }
      for (int i = 0; i < remain; i++) {
        *(dst++) = *(src++) * scale[j] + bias_data;
      }
    }
  }
}
template <>
void fill_bias_act_calib<int8_t>(int8_t* dout,
                                 const int32_t* din,
                                 const float* bias,
                                 const float* scale,
                                 int channel,
                                 int channel_size,
                                 bool flag_bias,
                                 const operators::ActivationParam* act_param) {
  int cnt_num = channel_size >> 4;
  int remain = channel_size & 15;
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t vmax = vdupq_n_f32(-127.f);
  if (act_param != nullptr && act_param->has_active) {
    float32x4_t vsix = vdupq_n_f32(act_param->Relu_clipped_coef);
    if (act_param->active_type == lite_api::ActivationType::kLeakyRelu) {
      vsix = vdupq_n_f32(act_param->Leaky_relu_alpha);
    }

    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
        for (int j = 0; j < channel; j++) {
          const float bias_data = flag_bias ? bias[j] : 0.f;
          const int32_t* src = din + j * channel_size;
          int8_t* dst = dout + j * channel_size;
          float32x4_t vscale_val = vdupq_n_f32(scale[j]);
          float32x4_t vbias = vdupq_n_f32(bias_data);
          int cnt = cnt_num;
          if (cnt_num > 0) {
#ifdef __aarch64__
            asm volatile(
                INT32_TO_FP32 FILL_RELU FP32_TO_INT8
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "v0",
                  "v1",
                  "v2",
                  "v3",
                  "v4",
                  "v5",
                  "v6",
                  "v7");
#else
            asm volatile(
                INT32_TO_FP32 FILL_RELU FP32_ROUND FP32_TO_INT8
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10",
                  "q11",
                  "q12",
                  "q13");
#endif
          }
          for (int i = 0; i < remain; i++) {
            float tmp = (*src * scale[j] + bias_data);
            tmp = tmp >= 0.f ? tmp : 0.f;
            dst[0] = saturate_cast<signed char>(roundf(tmp));
            src++;
            dst++;
          }
        }
        break;
      case lite_api::ActivationType::kRelu6:
        for (int j = 0; j < channel; j++) {
          const float bias_data = flag_bias ? bias[j] : 0.f;
          const int32_t* src = din + j * channel_size;
          int8_t* dst = dout + j * channel_size;
          float32x4_t vscale_val = vdupq_n_f32(scale[j]);
          float32x4_t vbias = vdupq_n_f32(bias_data);
          int cnt = cnt_num;
          if (cnt_num > 0) {
#ifdef __aarch64__
            asm volatile(
                INT32_TO_FP32 FILL_RELU FILL_RELU6 FP32_TO_INT8
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vsix] "w"(vsix),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "v0",
                  "v1",
                  "v2",
                  "v3",
                  "v4",
                  "v5",
                  "v6",
                  "v7");
#else
            asm volatile(
                INT32_TO_FP32 FILL_RELU FILL_RELU6 FP32_ROUND FP32_TO_INT8
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vsix] "w"(vsix),
                  [vscale_val] "w"(vscale_val)
                : "memory",
                  "cc",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10",
                  "q11",
                  "q12",
                  "q13");
#endif
          }
          for (int i = 0; i < remain; i++) {
            float tmp = (*src * scale[j] + bias_data);
            tmp = tmp >= 0.f ? tmp : 0.f;
            tmp = tmp <= act_param->Relu_clipped_coef
                      ? tmp
                      : act_param->Relu_clipped_coef;
            dst[0] = saturate_cast<signed char>(roundf(tmp));
            src++;
            dst++;
          }
        }
        break;
      case lite_api::ActivationType::kLeakyRelu:
        for (int j = 0; j < channel; j++) {
          const float bias_data = flag_bias ? bias[j] : 0.f;
          const int32_t* src = din + j * channel_size;
          int8_t* dst = dout + j * channel_size;
          float32x4_t vscale_val = vdupq_n_f32(scale[j]);
          float32x4_t vbias = vdupq_n_f32(bias_data);
          int cnt = cnt_num;
          if (cnt_num > 0) {
#ifdef __aarch64__
            asm volatile(
                INT32_TO_FP32 FILL_LEAKY_RELU FP32_MAX FP32_TO_INT8
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale] "w"(vsix),
                  [vscale_val] "w"(vscale_val),
                  [vmax] "w"(vmax)
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
#else
            asm volatile(
                INT32_TO_FP32 FILL_LEAKY_RELU FP32_ROUND FP32_MAX FP32_TO_INT8
                : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                : [vzero] "w"(vzero),
                  [vbias] "w"(vbias),
                  [vscale] "w"(vsix),
                  [vscale_val] "w"(vscale_val),
                  [vmax] "w"(vmax)
                : "memory",
                  "cc",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10",
                  "q11",
                  "q12",
                  "q13");
#endif
          }
          for (int i = 0; i < remain; i++) {
            float tmp = (*src * scale[j] + bias_data);
            tmp = tmp >= 0.f ? tmp : tmp * act_param->Leaky_relu_alpha;
            dst[0] = saturate_cast<signed char>(roundf(tmp));
            dst[0] = dst[0] < -127 ? -127 : dst[0];  // -127 - 127
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
      const float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vscale_val = vdupq_n_f32(scale[j]);
      float32x4_t vbias = vdupq_n_f32(bias_data);
      const int32_t* src = din + j * channel_size;
      int8_t* dst = dout + j * channel_size;
      int cnt = cnt_num;
      if (cnt > 0) {
#ifdef __aarch64__
        asm volatile(
            INT32_TO_FP32 FP32_MAX FP32_TO_INT8
            : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
            : [vbias] "w"(vbias), [vscale_val] "w"(vscale_val), [vmax] "w"(vmax)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
        asm volatile(INT32_TO_FP32 FP32_ROUND FP32_MAX FP32_TO_INT8
                     :
                     [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
                     : [vbias] "w"(vbias),
                       [vscale_val] "w"(vscale_val),
                       [vmax] "w"(vmax),
                       [vzero] "w"(vzero)
                     : "memory",
                       "cc",
                       "q3",
                       "q4",
                       "q5",
                       "q6",
                       "q7",
                       "q8",
                       "q9",
                       "q10",
                       "q11",
                       "q12",
                       "q13");
#endif
      }
      for (int i = 0; i < remain; i++) {
        float tmp = *(src++) * scale[j] + bias_data;
        dst[0] = saturate_cast<signed char>(roundf(tmp));
        dst[0] = dst[0] < -127 ? -127 : dst[0];  // -127 - 127
        dst++;
      }
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
