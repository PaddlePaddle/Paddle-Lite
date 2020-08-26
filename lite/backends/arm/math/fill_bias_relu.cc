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

template <>
void fill_bias_relu<float>(float* tensor,
                           const float* bias,
                           int channel,
                           int channel_size,
                           bool flag_bias,
                           bool flag_relu) {
  float* data = tensor;
  if (flag_relu) {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vbias = vdupq_n_f32(bias_data);
      float32x4_t vzero = vdupq_n_f32(0.f);
      int i = 0;
      for (; i < channel_size - 3; i += 4) {
        float32x4_t vdata = vld1q_f32(&data[i]);
        vdata = vaddq_f32(vdata, vbias);
        float32x4_t vmax = vmaxq_f32(vdata, vzero);
        vst1q_f32(data + i, vmax);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
        data[i] = data[i] > 0 ? data[i] : 0.f;
      }
      data += channel_size;
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vbias = vdupq_n_f32(bias_data);
      int i = 0;
      for (; i < channel_size - 3; i += 4) {
        float32x4_t vdata = vld1q_f32(&data[i]);
        vdata = vaddq_f32(vdata, vbias);
        vst1q_f32(data + i, vdata);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
      }
      data += channel_size;
    }
  }
}

template <>
void fill_bias_relu<int>(int* tensor,
                         const int* bias,
                         int channel,
                         int channel_size,
                         bool flag_bias,
                         bool flag_relu) {
  int* data = tensor;
  if (flag_relu) {
    for (int j = 0; j < channel; ++j) {
      int bias_data = flag_bias ? bias[j] : 0;
      int32x4_t vbias = vdupq_n_s32(bias_data);
      int32x4_t vzero = vdupq_n_s32(0);
      int i = 0;
      for (; i < channel_size - 7; i += 8) {
        int32x4_t vdata1 = vld1q_s32(data + i);
        int32x4_t vdata2 = vld1q_s32(data + i + 4);
        vdata1 = vaddq_s32(vdata1, vbias);
        vdata2 = vaddq_s32(vdata2, vbias);
        int32x4_t vmax1 = vmaxq_s32(vdata1, vzero);
        int32x4_t vmax2 = vmaxq_s32(vdata2, vzero);
        vst1q_s32(data + i, vmax1);
        vst1q_s32(data + i + 4, vmax2);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
        data[i] = data[i] > 0 ? data[i] : 0;
      }
      data += channel_size;
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      int bias_data = flag_bias ? bias[j] : 0;
      int32x4_t vbias = vdupq_n_s32(bias_data);
      int i = 0;
      for (; i < channel_size - 7; i += 8) {
        int32x4_t vdata1 = vld1q_s32(data + i);
        int32x4_t vdata2 = vld1q_s32(data + i + 4);
        vdata1 = vaddq_s32(vdata1, vbias);
        vdata2 = vaddq_s32(vdata2, vbias);
        vst1q_s32(data + i, vdata1);
        vst1q_s32(data + i + 4, vdata2);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
      }
      data += channel_size;
    }
  }
}
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
#define FILL_STORE                                       \
  "subs %w[cnt], %w[cnt], #1                    \n"      \
  "st1 {v0.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v1.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v2.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v3.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "bne  1b                                    \n"
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
  "vcge.f32 q13, q6, %q[vzero]   @ vcgeq_u32 \n" \
  "vmul.f32 q14, q6, %q[vscale]  @ vmulq_f32 \n" \
  "vbif q3, q8, q7               @ choose \n"    \
  "vbif q4, q10, q9              @ choose \n"    \
  "vbif q5, q12, q11             @ choose \n"    \
  "vbif q6, q14, q13             @ choose \n"
#define FILL_STORE                                          \
  "subs %[cnt], #1                                \n"       \
  "vst1.32 {d6-d7}, [%[dout_ptr]]!       @ vst1q_f32()  \n" \
  "vst1.32 {d8-d9}, [%[dout_ptr]]!       @ vst1q_f32()  \n" \
  "vst1.32 {d10-d11}, [%[dout_ptr]]!     @ vst1q_f32()  \n" \
  "vst1.32 {d12-d13}, [%[dout_ptr]]!     @ vst1q_f32()  \n" \
  "bne  1b                                    \n"
#endif
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
    float32x4_t vsix = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t vscale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
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
        break;
      case lite_api::ActivationType::kRelu6:
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
        break;
      case lite_api::ActivationType::kLeakyRelu:
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
        break;
      default:
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
      }
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
