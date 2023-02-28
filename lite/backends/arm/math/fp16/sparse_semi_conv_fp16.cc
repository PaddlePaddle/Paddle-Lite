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

#include "lite/backends/arm/math/fp16/sparse_semi_conv_fp16.h"
#include <arm_neon.h>
#include <vector>
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#define INIT_SEMI_CONV_PARAM(Dtype_i, Dtype_o, start_w)                    \
  auto act_param = param.activation_param;                                 \
  auto act_type = act_param.active_type;                                   \
  volatile float16_t alpha = 0.f;                                          \
  int flag_act = 0x00; /* relu: 1, relu6: 2, leakey: 3  */                 \
  float16_t hs_param[24] = {0.f};                                          \
  if (act_param.has_active) {                                              \
    if (act_type == lite_api::ActivationType::kRelu) {                     \
      flag_act = 0x01;                                                     \
    } else if (act_type == lite_api::ActivationType::kRelu6) {             \
      flag_act = 0x02;                                                     \
      alpha = static_cast<float16_t>(act_param.Relu_clipped_coef);         \
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {         \
      flag_act = 0x03;                                                     \
      alpha = static_cast<float16_t>(act_param.Leaky_relu_alpha);          \
    } else if (act_type == lite_api::ActivationType::kHardSwish) {         \
      flag_act = 0x04;                                                     \
      for (int i = 0; i < 8; i++) {                                        \
        hs_param[i] = static_cast<float16_t>(act_param.hard_swish_offset); \
        hs_param[i + 8] =                                                  \
            static_cast<float16_t>(1.0 / act_param.hard_swish_scale);      \
        hs_param[i + 16] =                                                 \
            static_cast<float16_t>(act_param.hard_swish_threshold);        \
      }                                                                    \
    }                                                                      \
  }                                                                        \
  int flag_bias = (bias != nullptr) ? 1 : 0;                               \
  size_t mc = N * sizeof(Dtype_i);                                         \
  size_t nc = M;                                                           \
  size_t output_stride = N * sizeof(Dtype_o);                              \
  size_t pair_num = M / 2;                                                 \
  size_t lave_num = M % 2;

#define GET_SEMI_PARAM_TABLE(Dtype_i, Dtype_o)                                 \
  Dtype_o* out_ptr1 =                                                          \
      reinterpret_cast<Dtype_o*>((uintptr_t)output + output_stride * 2 * i);   \
  Dtype_o* out_ptr2 = reinterpret_cast<Dtype_o*>((uintptr_t)output +           \
                                                 output_stride * (2 * i + 1)); \
  const Dtype_i* cur_w = A;                                                    \
  uint32_t nnz = nidx_nnzmap[i];                                               \
  const Dtype_i* cur_b = B;                                                    \
  const int32_t* dmap = widx_dmap;                                             \
  if (i != 0) {                                                                \
    cur_w = A + nidx_nnzmap[i - 1] * 2;                                        \
    nnz = nidx_nnzmap[i] - nidx_nnzmap[i - 1];                                 \
    cur_b += ((nidx_nnzmap[i - 1] == 0)                                        \
                  ? 0                                                          \
                  : (widx_dmap[nidx_nnzmap[i - 1] - 1] / sizeof(Dtype_i)));    \
    dmap = widx_dmap + nidx_nnzmap[i - 1];                                     \
  }                                                                            \
  const float16_t* pbias = (bias != nullptr) ? (bias + i * 2) : bias_zero;

#define GET_UNSTRUCT_PARAM_TABLE(Dtype_i, Dtype_o)                             \
  Dtype_o* out_ptr = reinterpret_cast<Dtype_o*>((uintptr_t)output +            \
                                                output_stride * 2 * pair_num); \
  const Dtype_i* cur_w = A;                                                    \
  uint32_t nnz = nidx_nnzmap[pair_num];                                        \
  const Dtype_i* cur_b = B;                                                    \
  const int32_t* dmap = widx_dmap;                                             \
  if (pair_num != 0) {                                                         \
    cur_w = A + nidx_nnzmap[pair_num - 1] * 2;                                 \
    nnz = nidx_nnzmap[pair_num] - nidx_nnzmap[pair_num - 1];                   \
    cur_b +=                                                                   \
        ((nidx_nnzmap[pair_num - 1] == 0)                                      \
             ? 0                                                               \
             : (widx_dmap[nidx_nnzmap[pair_num - 1] - 1] / sizeof(Dtype_i)));  \
    dmap = widx_dmap + nidx_nnzmap[pair_num - 1];                              \
  }                                                                            \
  float16_t vbias = (bias != nullptr) ? bias[pair_num * 2] : 0.0;

#define COMPUTE_ACT_NEON_TWO_V8_F32                                \
  if (flag_act == 1) {                                             \
    vacc01n0 = vacc01n0 > 0.f ? vacc01n0 : 0.f;                    \
    vacc01n1 = vacc01n1 > 0.f ? vacc01n1 : 0.f;                    \
  } else if (flag_act == 0) {                                      \
  } else if (flag_act == 2) {                                      \
    vacc01n0 = vacc01n0 > 0.f ? vacc01n0 : 0.f;                    \
    vacc01n1 = vacc01n1 > 0.f ? vacc01n1 : 0.f;                    \
    vacc01n0 = vacc01n0 < alpha ? vacc01n0 : alpha;                \
    vacc01n1 = vacc01n1 < alpha ? vacc01n1 : alpha;                \
  } else if (flag_act == 3) {                                      \
    vacc01n0 = vacc01n0 > 0.f ? vacc01n0 : alpha * vacc01n0;       \
    vacc01n1 = vacc01n1 > 0.f ? vacc01n1 : alpha * vacc01n1;       \
  } else if (flag_act == 4) {                                      \
    auto tmp0_0 = vacc01n0 + act_param.hard_swish_offset;          \
    auto tmp1_0 = vacc01n1 + act_param.hard_swish_offset;          \
    auto tmp0_1 = vacc01n0 * act_param.hard_swish_scale;           \
    auto tmp1_1 = vacc01n1 * act_param.hard_swish_scale;           \
    tmp0_0 = tmp0_0 > 0.f ? tmp0_0 : 0.f;                          \
    tmp1_0 = tmp1_0 > 0.f ? tmp1_0 : 0.f;                          \
    tmp0_0 = tmp0_0 < act_param.hard_swish_threshold               \
                 ? tmp0_0                                          \
                 : act_param.hard_swish_threshold;                 \
    tmp1_0 = tmp1_0 < act_param.hard_swish_threshold               \
                 ? tmp1_0                                          \
                 : act_param.hard_swish_threshold;                 \
    vacc01n0 = tmp0_0 * tmp0_1;                                    \
    vacc01n1 = tmp1_0 * tmp1_1;                                    \
  } else {                                                         \
    LOG(FATAL) << "This act: " << flag_act << " doesn't support!"; \
  }

#define COMPUTE_ACT_NEON_ONE_V8_F32                                \
  if (flag_act == 1) {                                             \
    vacc01n0 = vacc01n0 > 0.f ? vacc01n0 : 0.f;                    \
  } else if (flag_act == 0) {                                      \
  } else if (flag_act == 2) {                                      \
    vacc01n0 = vacc01n0 > 0.f ? vacc01n0 : 0.f;                    \
    vacc01n0 = vacc01n0 < alpha ? vacc01n0 : alpha;                \
  } else if (flag_act == 3) {                                      \
    vacc01n0 = vacc01n0 > 0.f ? vacc01n0 : alpha * vacc01n0;       \
  } else if (flag_act == 4) {                                      \
    auto tmp0_0 = vacc01n0 + act_param.hard_swish_offset;          \
    auto tmp0_1 = vacc01n0 * act_param.hard_swish_scale;           \
    tmp0_0 = tmp0_0 > 0.f ? tmp0_0 : 0.f;                          \
    tmp0_0 = tmp0_0 < act_param.hard_swish_threshold               \
                 ? tmp0_0                                          \
                 : act_param.hard_swish_threshold;                 \
    vacc01n0 = tmp0_0 * tmp0_1;                                    \
  } else {                                                         \
    LOG(FATAL) << "This act: " << flag_act << " doesn't support!"; \
  }

#ifdef __aarch64__
#define SPARSE_F16_F16_W48_SEMI2_V8_KERNEL  \
  "ldr     q0,   [%[bias_ptr]]\n"           \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "dup     v20.8h,  v0.h[0]\n"              \
  "dup     v21.8h,  v0.h[0]\n"              \
  "dup     v22.8h,  v0.h[0]\n"              \
  "dup     v23.8h,  v0.h[0]\n"              \
  "dup     v24.8h,  v0.h[0]\n"              \
  "dup     v25.8h,  v0.h[0]\n"              \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v26.8h,  v0.h[1]\n"              \
  "dup     v27.8h,  v0.h[1]\n"              \
  "dup     v28.8h,  v0.h[1]\n"              \
  "dup     v29.8h,  v0.h[1]\n"              \
  "dup     v30.8h,  v0.h[1]\n"              \
  "dup     v31.8h,  v0.h[1]\n"              \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldp   q1, q2, [%[b_ptr]]\n"              \
  "ldr   w1, [%[widx_dmap]],  #4\n"         \
  "add   %[a_ptr],  %[a_ptr], #4\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "ldp   q3, q4, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v26.8h,  v1.8h,  v0.h[1]\n"        \
  "fmla  v21.8h,  v2.8h,  v0.h[0]\n"        \
  "fmla  v27.8h,  v2.8h,  v0.h[1]\n"        \
  "ldp   q5, q6, [%[b_ptr], #64]\n"         \
  "fmla  v22.8h,  v3.8h,  v0.h[0]\n"        \
  "fmla  v28.8h,  v3.8h,  v0.h[1]\n"        \
  "fmla  v23.8h,  v4.8h,  v0.h[0]\n"        \
  "fmla  v29.8h,  v4.8h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v24.8h,  v5.8h,  v0.h[0]\n"        \
  "fmla  v30.8h,  v5.8h,  v0.h[1]\n"        \
  "fmla  v25.8h,  v6.8h,  v0.h[0]\n"        \
  "fmla  v31.8h,  v6.8h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W48_SEMI1_V8_KERNEL  \
  "dup     v20.8h,  %w[vbias]\n"            \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v22.8h,  v20.h[0]\n"             \
  "dup     v23.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v24.8h,  v20.h[0]\n"             \
  "dup     v25.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldp   q1, q2, [%[b_ptr]]\n"              \
  "ldr   w1, [%[widx_dmap]],  #4\n"         \
  "add   %[a_ptr],  %[a_ptr], #2\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "ldp   q3, q4, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v21.8h,  v2.8h,  v0.h[0]\n"        \
  "ldp   q5, q6, [%[b_ptr], #64]\n"         \
  "fmla  v22.8h,  v3.8h,  v0.h[0]\n"        \
  "fmla  v23.8h,  v4.8h,  v0.h[0]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v24.8h,  v5.8h,  v0.h[0]\n"        \
  "fmla  v25.8h,  v6.8h,  v0.h[0]\n"        \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W32_SEMI2_V8_KERNEL  \
  "ldr     q0,   [%[bias_ptr]]\n"           \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "dup     v20.8h,  v0.h[0]\n"              \
  "dup     v21.8h,  v0.h[0]\n"              \
  "dup     v22.8h,  v0.h[0]\n"              \
  "dup     v23.8h,  v0.h[0]\n"              \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v26.8h,  v0.h[1]\n"              \
  "dup     v27.8h,  v0.h[1]\n"              \
  "dup     v28.8h,  v0.h[1]\n"              \
  "dup     v29.8h,  v0.h[1]\n"              \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldp   q1, q2, [%[b_ptr]]\n"              \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "add   %[a_ptr],  %[a_ptr], #4\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "ldp   q3, q4, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v26.8h,  v1.8h,  v0.h[1]\n"        \
  "fmla  v21.8h,  v2.8h,  v0.h[0]\n"        \
  "fmla  v27.8h,  v2.8h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v22.8h,  v3.8h,  v0.h[0]\n"        \
  "fmla  v28.8h,  v3.8h,  v0.h[1]\n"        \
  "fmla  v23.8h,  v4.8h,  v0.h[0]\n"        \
  "fmla  v29.8h,  v4.8h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W32_SEMI1_V8_KERNEL  \
  "dup     v20.8h,  %w[vbias]\n"            \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v22.8h,  v20.h[0]\n"             \
  "dup     v23.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldp   q1, q2, [%[b_ptr]]\n"              \
  "ldr   w1, [%[widx_dmap]],  #4\n"         \
  "add   %[a_ptr],  %[a_ptr], #2\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "ldp   q3, q4, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v21.8h,  v2.8h,  v0.h[0]\n"        \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v22.8h,  v3.8h,  v0.h[0]\n"        \
  "fmla  v23.8h,  v4.8h,  v0.h[0]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W16_SEMI2_V8_KERNEL  \
  "ldr     q0,   [%[bias_ptr]]\n"           \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "dup     v20.8h,  v0.h[0]\n"              \
  "dup     v21.8h,  v0.h[0]\n"              \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v26.8h,  v0.h[1]\n"              \
  "dup     v27.8h,  v0.h[1]\n"              \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldp   q1, q2, [%[b_ptr]]\n"              \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "add   %[a_ptr],  %[a_ptr], #4\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v26.8h,  v1.8h,  v0.h[1]\n"        \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v21.8h,  v2.8h,  v0.h[0]\n"        \
  "fmla  v27.8h,  v2.8h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W16_SEMI1_V8_KERNEL  \
  "dup     v20.8h,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldp   q1, q2, [%[b_ptr]]\n"              \
  "ldr   w1, [%[widx_dmap]],  #4\n"         \
  "add   %[a_ptr],  %[a_ptr], #2\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v21.8h,  v2.8h,  v0.h[0]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W8_SEMI2_V8_KERNEL   \
  "ldr     q0,   [%[bias_ptr]]\n"           \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v20.8h,  v0.h[0]\n"              \
  "dup     v26.8h,  v0.h[1]\n"              \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldr   q1, [%[b_ptr]]\n"                  \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "add   %[a_ptr],  %[a_ptr], #4\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "fmla  v26.8h,  v1.8h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W8_SEMI1_V8_KERNEL   \
  "dup     v20.8h,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldr   q1, [%[b_ptr]]\n"                  \
  "ldr   w1, [%[widx_dmap]],  #4\n"         \
  "add   %[a_ptr],  %[a_ptr], #2\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v20.8h,  v1.8h,  v0.h[0]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W4_SEMI2_V8_KERNEL   \
  "ldr     q0,   [%[bias_ptr]]\n"           \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v20.4h,  v0.h[0]\n"              \
  "dup     v26.4h,  v0.h[1]\n"              \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldr   d1, [%[b_ptr]]\n"                  \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "add   %[a_ptr],  %[a_ptr], #4\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v20.4h,  v1.4h,  v0.h[0]\n"        \
  "fmla  v26.4h,  v1.4h,  v0.h[1]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W4_SEMI1_V8_KERNEL   \
  "dup     v20.4h,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz     %w[k],    1f\n" /* main loop*/   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]]\n"                  \
  "ldr   d1, [%[b_ptr]]\n"                  \
  "ldr   w1, [%[widx_dmap]],  #4\n"         \
  "add   %[a_ptr],  %[a_ptr], #2\n"         \
  "subs  %w[k],   %w[k],   #1\n"            \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla  v20.4h,  v1.4h,  v0.h[0]\n"        \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "bne     0b\n"                            \
  "1:\n"

#define SPARSE_F16_F16_W48_SEMI2_V8_RELU \
  /* do relu */                          \
  "cmp    %w[vflag_act],    #0\n"        \
  "beq   9f                     \n"      \
  "cmp    %w[vflag_act],    #1\n"        \
  "bne   10f                     \n"     \
  "movi   v10.8h, #0\n"                  \
  "fmax   v20.8h, v20.8h, v10.8h\n"      \
  "fmax   v21.8h, v21.8h, v10.8h\n"      \
  "fmax   v22.8h, v22.8h, v10.8h\n"      \
  "fmax   v23.8h, v23.8h, v10.8h\n"      \
  "fmax   v24.8h, v24.8h, v10.8h\n"      \
  "fmax   v25.8h, v25.8h, v10.8h\n"      \
  "fmax   v26.8h, v26.8h, v10.8h\n"      \
  "fmax   v27.8h, v27.8h, v10.8h\n"      \
  "fmax   v28.8h, v28.8h, v10.8h\n"      \
  "fmax   v29.8h, v29.8h, v10.8h\n"      \
  "fmax   v30.8h, v30.8h, v10.8h\n"      \
  "fmax   v31.8h, v31.8h, v10.8h\n"      \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_SEMI1_V8_RELU \
  /* do relu */                          \
  "cmp    %w[vflag_act],    #0\n"        \
  "beq   9f                     \n"      \
  "cmp    %w[vflag_act],    #1\n"        \
  "bne   10f                     \n"     \
  "movi   v10.8h, #0\n"                  \
  "fmax   v20.8h, v20.8h, v10.8h\n"      \
  "fmax   v21.8h, v21.8h, v10.8h\n"      \
  "fmax   v22.8h, v22.8h, v10.8h\n"      \
  "fmax   v23.8h, v23.8h, v10.8h\n"      \
  "fmax   v24.8h, v24.8h, v10.8h\n"      \
  "fmax   v25.8h, v25.8h, v10.8h\n"      \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI2_V8_RELU \
  /* do relu */                          \
  "cmp    %w[vflag_act],    #0\n"        \
  "beq   9f                     \n"      \
  "cmp    %w[vflag_act],    #1\n"        \
  "bne   10f                     \n"     \
  "movi   v10.8h, #0\n"                  \
  "fmax   v20.8h, v20.8h, v10.8h\n"      \
  "fmax   v21.8h, v21.8h, v10.8h\n"      \
  "fmax   v22.8h, v22.8h, v10.8h\n"      \
  "fmax   v23.8h, v23.8h, v10.8h\n"      \
  "fmax   v26.8h, v26.8h, v10.8h\n"      \
  "fmax   v27.8h, v27.8h, v10.8h\n"      \
  "fmax   v28.8h, v28.8h, v10.8h\n"      \
  "fmax   v29.8h, v29.8h, v10.8h\n"      \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI1_V8_RELU \
  /* do relu */                          \
  "cmp    %w[vflag_act],    #0\n"        \
  "beq   9f                     \n"      \
  "cmp    %w[vflag_act],    #1\n"        \
  "bne   10f                     \n"     \
  "movi   v9.8h, #0\n"                   \
  "fmax   v20.8h, v20.8h, v9.8h\n"       \
  "fmax   v21.8h, v21.8h, v9.8h\n"       \
  "fmax   v22.8h, v22.8h, v9.8h\n"       \
  "fmax   v23.8h, v23.8h, v9.8h\n"       \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI2_V8_RELU \
  /* do relu */                          \
  "cmp    %w[vflag_act],    #0\n"        \
  "beq   9f                     \n"      \
  "cmp    %w[vflag_act],    #1\n"        \
  "bne   10f                     \n"     \
  "movi   v10.8h, #0\n"                  \
  "fmax   v20.8h, v20.8h, v10.8h\n"      \
  "fmax   v21.8h, v21.8h, v10.8h\n"      \
  "fmax   v26.8h, v26.8h, v10.8h\n"      \
  "fmax   v27.8h, v27.8h, v10.8h\n"      \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI1_V8_RELU \
  /* do relu */                          \
  "cmp    %w[vflag_act],    #0\n"        \
  "beq   9f                     \n"      \
  "cmp    %w[vflag_act],    #1\n"        \
  "bne   10f                     \n"     \
  "movi   v9.8h, #0\n"                   \
  "fmax   v20.8h, v20.8h, v9.8h\n"       \
  "fmax   v21.8h, v21.8h, v9.8h\n"       \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI2_V8_RELU \
  /* do relu */                         \
  "cmp    %w[vflag_act],    #0\n"       \
  "beq   9f                     \n"     \
  "cmp    %w[vflag_act],    #1\n"       \
  "bne   10f                     \n"    \
  "movi   v10.8h, #0\n"                 \
  "fmax   v20.8h, v20.8h, v10.8h\n"     \
  "fmax   v26.8h, v26.8h, v10.8h\n"     \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI1_V8_RELU \
  /* do relu */                         \
  "cmp    %w[vflag_act],    #0\n"       \
  "beq   9f                     \n"     \
  "cmp    %w[vflag_act],    #1\n"       \
  "bne   10f                     \n"    \
  "movi   v9.8h, #0\n"                  \
  "fmax   v20.8h, v20.8h, v9.8h\n"      \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI2_V8_RELU \
  /* do relu */                         \
  "cmp    %w[vflag_act],    #0\n"       \
  "beq   9f                     \n"     \
  "cmp    %w[vflag_act],    #1\n"       \
  "bne   10f                     \n"    \
  "movi   v10.4h, #0\n"                 \
  "fmax   v20.4h, v20.4h, v10.4h\n"     \
  "fmax   v26.4h, v26.4h, v10.4h\n"     \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI1_V8_RELU \
  /* do relu */                         \
  "cmp    %w[vflag_act],    #0\n"       \
  "beq   9f                     \n"     \
  "cmp    %w[vflag_act],    #1\n"       \
  "bne   10f                     \n"    \
  "movi   v9.4h, #0\n"                  \
  "fmax   v20.4h, v20.4h, v9.4h\n"      \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_SEMI2_V8_RELU6 \
  /* do relu6 */                          \
  "10: \n"                                \
  "cmp   %w[vflag_act],  #2       \n"     \
  "bne   11f                     \n"      \
  "movi   v0.8h, #0\n"                    \
  "dup    v1.8h,  %w[valpha]\n"           \
  "fmax   v20.8h, v20.8h, v0.8h\n"        \
  "fmax   v21.8h, v21.8h, v0.8h\n"        \
  "fmax   v22.8h, v22.8h, v0.8h\n"        \
  "fmax   v23.8h, v23.8h, v0.8h\n"        \
  "fmax   v24.8h, v24.8h, v0.8h\n"        \
  "fmax   v25.8h, v25.8h, v0.8h\n"        \
  "fmax   v26.8h, v26.8h, v0.8h\n"        \
  "fmax   v27.8h, v27.8h, v0.8h\n"        \
  "fmax   v28.8h, v28.8h, v0.8h\n"        \
  "fmax   v29.8h, v29.8h, v0.8h\n"        \
  "fmax   v30.8h, v30.8h, v0.8h\n"        \
  "fmax   v31.8h, v31.8h, v0.8h\n"        \
  "fmin   v20.8h, v20.8h, v1.8h\n"        \
  "fmin   v21.8h, v21.8h, v1.8h\n"        \
  "fmin   v22.8h, v22.8h, v1.8h\n"        \
  "fmin   v23.8h, v23.8h, v1.8h\n"        \
  "fmin   v24.8h, v24.8h, v1.8h\n"        \
  "fmin   v25.8h, v25.8h, v1.8h\n"        \
  "fmin   v26.8h, v26.8h, v1.8h\n"        \
  "fmin   v27.8h, v27.8h, v1.8h\n"        \
  "fmin   v28.8h, v28.8h, v1.8h\n"        \
  "fmin   v29.8h, v29.8h, v1.8h\n"        \
  "fmin   v30.8h, v30.8h, v1.8h\n"        \
  "fmin   v31.8h, v31.8h, v1.8h\n"        \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_SEMI1_V8_RELU6 \
  /* do relu6 */                          \
  "10: \n"                                \
  "cmp   %w[vflag_act],  #2       \n"     \
  "bne   11f                     \n"      \
  "movi   v0.8h, #0\n"                    \
  "dup    v1.8h,  %w[valpha]\n"           \
  "fmax   v20.8h, v20.8h, v0.8h\n"        \
  "fmax   v21.8h, v21.8h, v0.8h\n"        \
  "fmax   v22.8h, v22.8h, v0.8h\n"        \
  "fmax   v23.8h, v23.8h, v0.8h\n"        \
  "fmax   v24.8h, v24.8h, v0.8h\n"        \
  "fmax   v25.8h, v25.8h, v0.8h\n"        \
  "fmin   v20.8h, v20.8h, v1.8h\n"        \
  "fmin   v21.8h, v21.8h, v1.8h\n"        \
  "fmin   v22.8h, v22.8h, v1.8h\n"        \
  "fmin   v23.8h, v23.8h, v1.8h\n"        \
  "fmin   v24.8h, v24.8h, v1.8h\n"        \
  "fmin   v25.8h, v25.8h, v1.8h\n"        \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI2_V8_RELU6 \
  /* do relu6 */                          \
  "10: \n"                                \
  "cmp   %w[vflag_act],  #2       \n"     \
  "bne   11f                     \n"      \
  "movi   v0.8h, #0\n"                    \
  "dup    v1.8h,  %w[valpha]\n"           \
  "fmax   v20.8h, v20.8h, v0.8h\n"        \
  "fmax   v21.8h, v21.8h, v0.8h\n"        \
  "fmax   v22.8h, v22.8h, v0.8h\n"        \
  "fmax   v23.8h, v23.8h, v0.8h\n"        \
  "fmax   v26.8h, v26.8h, v0.8h\n"        \
  "fmax   v27.8h, v27.8h, v0.8h\n"        \
  "fmax   v28.8h, v28.8h, v0.8h\n"        \
  "fmax   v29.8h, v29.8h, v0.8h\n"        \
  "fmin   v20.8h, v20.8h, v1.8h\n"        \
  "fmin   v21.8h, v21.8h, v1.8h\n"        \
  "fmin   v22.8h, v22.8h, v1.8h\n"        \
  "fmin   v23.8h, v23.8h, v1.8h\n"        \
  "fmin   v26.8h, v26.8h, v1.8h\n"        \
  "fmin   v27.8h, v27.8h, v1.8h\n"        \
  "fmin   v28.8h, v28.8h, v1.8h\n"        \
  "fmin   v29.8h, v29.8h, v1.8h\n"        \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI1_V8_RELU6 \
  /* do relu6 */                          \
  "10: \n"                                \
  "cmp   %w[vflag_act],  #2       \n"     \
  "bne   11f                     \n"      \
  "movi   v0.8h, #0\n"                    \
  "dup    v1.8h,  %w[valpha]\n"           \
  "fmax   v20.8h, v20.8h, v0.8h\n"        \
  "fmax   v21.8h, v21.8h, v0.8h\n"        \
  "fmax   v22.8h, v22.8h, v0.8h\n"        \
  "fmax   v23.8h, v23.8h, v0.8h\n"        \
  "fmin   v20.8h, v20.8h, v1.8h\n"        \
  "fmin   v21.8h, v21.8h, v1.8h\n"        \
  "fmin   v22.8h, v22.8h, v1.8h\n"        \
  "fmin   v23.8h, v23.8h, v1.8h\n"        \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI2_V8_RELU6 \
  /* do relu6 */                          \
  "10: \n"                                \
  "cmp   %w[vflag_act],  #2       \n"     \
  "bne   11f                     \n"      \
  "movi   v0.8h, #0\n"                    \
  "dup    v1.8h,  %w[valpha]\n"           \
  "fmax   v20.8h, v20.8h, v0.8h\n"        \
  "fmax   v21.8h, v21.8h, v0.8h\n"        \
  "fmax   v26.8h, v26.8h, v0.8h\n"        \
  "fmax   v27.8h, v27.8h, v0.8h\n"        \
  "fmin   v20.8h, v20.8h, v1.8h\n"        \
  "fmin   v21.8h, v21.8h, v1.8h\n"        \
  "fmin   v26.8h, v26.8h, v1.8h\n"        \
  "fmin   v27.8h, v27.8h, v1.8h\n"        \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI1_V8_RELU6 \
  /* do relu6 */                          \
  "10: \n"                                \
  "cmp   %w[vflag_act],  #2       \n"     \
  "bne   11f                     \n"      \
  "movi   v0.8h, #0\n"                    \
  "dup    v1.8h,  %w[valpha]\n"           \
  "fmax   v20.8h, v20.8h, v0.8h\n"        \
  "fmax   v21.8h, v21.8h, v0.8h\n"        \
  "fmin   v20.8h, v20.8h, v1.8h\n"        \
  "fmin   v21.8h, v21.8h, v1.8h\n"        \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI2_V8_RELU6 \
  /* do relu6 */                         \
  "10: \n"                               \
  "cmp   %w[vflag_act],  #2       \n"    \
  "bne   11f                     \n"     \
  "movi   v0.8h, #0\n"                   \
  "dup    v1.8h,  %w[valpha]\n"          \
  "fmax   v20.8h, v20.8h, v0.8h\n"       \
  "fmax   v26.8h, v26.8h, v0.8h\n"       \
  "fmin   v20.8h, v20.8h, v1.8h\n"       \
  "fmin   v26.8h, v26.8h, v1.8h\n"       \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI1_V8_RELU6 \
  /* do relu6 */                         \
  "10: \n"                               \
  "cmp   %w[vflag_act],  #2       \n"    \
  "bne   11f                     \n"     \
  "movi   v0.8h, #0\n"                   \
  "dup    v1.8h,  %w[valpha]\n"          \
  "fmax   v20.8h, v20.8h, v0.8h\n"       \
  "fmin   v20.8h, v20.8h, v1.8h\n"       \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI2_V8_RELU6 \
  /* do relu6 */                         \
  "10: \n"                               \
  "cmp   %w[vflag_act],  #2       \n"    \
  "bne   11f                     \n"     \
  "movi   v0.4h, #0\n"                   \
  "dup    v1.4h,  %w[valpha]\n"          \
  "fmax   v20.4h, v20.4h, v0.4h\n"       \
  "fmax   v26.4h, v26.4h, v0.4h\n"       \
  "fmin   v20.4h, v20.4h, v1.4h\n"       \
  "fmin   v26.4h, v26.4h, v1.4h\n"       \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI1_V8_RELU6 \
  /* do relu6 */                         \
  "10: \n"                               \
  "cmp   %w[vflag_act],  #2       \n"    \
  "bne   11f                     \n"     \
  "movi   v0.4h, #0\n"                   \
  "dup    v1.4h,  %w[valpha]\n"          \
  "fmax   v20.4h, v20.4h, v0.4h\n"       \
  "fmin   v20.4h, v20.4h, v1.4h\n"       \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_SEMI2_V8_LEAKY_RELU                      \
  /* do relu */                                                     \
  "11: \n"                                                          \
  "cmp    %w[vflag_act],  #3       \n"                              \
  "bne    12f                     \n"                               \
  "movi   v0.8h, #0\n"                      /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"             /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v4.8h,    v21.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v5.8h,    v21.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v6.8h,    v22.8h,   v0.8h   \n"   /* vcgeq_f32 */         \
  "fmul   v7.8h,    v22.8h,   v1.8h   \n"   /* vmulq_f32 */         \
  "fcmge  v8.8h,    v23.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v9.8h,    v23.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v10.8h,   v24.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v11.8h,   v24.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v12.8h,   v25.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v13.8h,   v25.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "bif    v20.16b,  v3.16b,   v2.16b  \n"   /* choose*/             \
  "bif    v21.16b,  v5.16b,   v4.16b  \n"   /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v24.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v25.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "fcmge  v2.8h,    v26.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v3.8h,    v26.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v4.8h,    v27.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v5.8h,    v27.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v6.8h,    v28.8h,   v0.8h   \n"   /* vcgeq_f32 */         \
  "fmul   v7.8h,    v28.8h,   v1.8h   \n"   /* vmulq_f32 */         \
  "fcmge  v8.8h,    v29.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v9.8h,    v29.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v10.8h,   v30.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v11.8h,   v30.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v12.8h,   v31.8h,    v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v13.8h,   v31.8h,    v1.8h   \n"  /* vmulq_f32 */         \
  "bif    v26.16b,  v3.16b,   v2.16b  \n"   /* choose*/             \
  "bif    v27.16b,  v5.16b,   v4.16b  \n"   /* choose*/             \
  "bif    v28.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v29.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v30.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v31.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_SEMI1_V8_LEAKY_RELU                     \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v4.8h,    v21.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v21.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v6.8h,    v22.8h,   v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v7.8h,    v22.8h,   v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v8.8h,    v23.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v9.8h,    v23.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "fcmge  v2.8h,    v24.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v24.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "fcmge  v4.8h,    v25.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v25.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v24.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v25.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI2_V8_LEAKY_RELU                     \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v4.8h,    v21.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v21.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v6.8h,    v22.8h,   v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v7.8h,    v22.8h,   v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v8.8h,    v23.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v9.8h,    v23.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,  v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v21.16b,  v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "fcmge  v2.8h,    v26.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v26.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v4.8h,    v27.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v27.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v6.8h,    v28.8h,   v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v7.8h,    v28.8h,   v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v8.8h,    v29.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v9.8h,    v29.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v26.16b,  v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v27.16b,  v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v28.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v29.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI1_V8_LEAKY_RELU                     \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v4.8h,    v21.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v21.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v6.8h,    v22.8h,   v0.8h   \n"  /* vcgeq_f32 */         \
  "fmul   v7.8h,    v22.8h,   v1.8h   \n"  /* vmulq_f32 */         \
  "fcmge  v8.8h,    v23.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v9.8h,    v23.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v21.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI2_V8_LEAKY_RELU                     \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v4.8h,    v21.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v21.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v6.8h,    v26.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v7.8h,    v26.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v8.8h,    v27.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v9.8h,    v27.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,  v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v21.16b,  v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v26.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v27.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI1_V8_LEAKY_RELU                     \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v4.8h,    v21.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v5.8h,    v21.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v21.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI2_V8_LEAKY_RELU                      \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "fcmge  v6.8h,    v26.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v7.8h,    v26.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,  v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v26.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI1_V8_LEAKY_RELU                      \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.8h, #0\n"                     /* for relu6 */         \
  "dup    v1.8h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.8h,    v20.8h,    v0.8h   \n" /* vcgeq_f32 */         \
  "fmul   v3.8h,    v20.8h,    v1.8h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI2_V8_LEAKY_RELU                      \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.4h, #0\n"                     /* for relu6 */         \
  "dup    v1.4h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4h,    v20.4h,    v0.4h   \n" /* vcgeq_f32 */         \
  "fmul   v3.4h,    v20.4h,    v1.4h   \n" /* vmulq_f32 */         \
  "fcmge  v6.4h,    v26.4h,    v0.4h   \n" /* vcgeq_f32 */         \
  "fmul   v7.4h,    v26.4h,    v1.4h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,  v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v26.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI1_V8_LEAKY_RELU                      \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "cmp    %w[vflag_act],  #3       \n"                             \
  "bne    12f                     \n"                              \
  "movi   v0.4h, #0\n"                     /* for relu6 */         \
  "dup    v1.4h,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4h,    v20.4h,    v0.4h   \n" /* vcgeq_f32 */         \
  "fmul   v3.4h,    v20.4h,    v1.4h   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_SEMI2_V8_HARD_SWISH                   \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v4.8h,  v20.8h, v1.8h        \n"                       \
  "fadd   v6.8h,  v21.8h, v1.8h        \n"                       \
  "fadd   v8.8h,  v22.8h, v1.8h        \n"                       \
  "fadd   v10.8h,  v23.8h, v1.8h        \n"                      \
  "fadd   v12.8h,  v24.8h, v1.8h        \n"                      \
  "fadd   v14.8h,  v25.8h, v1.8h        \n"                      \
  "fadd   v16.8h,  v26.8h, v1.8h        \n"                      \
  "fadd   v18.8h,  v27.8h, v1.8h        \n"                      \
  "fmul   v5.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v7.8h,   v21.8h, v2.8h        \n"                      \
  "fmul   v9.8h,   v22.8h, v2.8h        \n"                      \
  "fmul   v11.8h,  v23.8h, v2.8h        \n"                      \
  "fmul   v13.8h,  v24.8h, v2.8h        \n"                      \
  "fmul   v15.8h,  v25.8h, v2.8h        \n"                      \
  "fmul   v17.8h,  v26.8h, v2.8h        \n"                      \
  "fmul   v19.8h,  v27.8h, v2.8h        \n"                      \
  "fmax   v4.8h,  v4.8h, v0.8h        \n"                        \
  "fmax   v6.8h,  v6.8h, v0.8h        \n"                        \
  "fmax   v8.8h,  v8.8h, v0.8h        \n"                        \
  "fmax   v10.8h, v10.8h, v0.8h       \n"                        \
  "fmax   v12.8h, v12.8h, v0.8h       \n"                        \
  "fmax   v14.8h, v14.8h, v0.8h       \n"                        \
  "fmax   v16.8h, v16.8h, v0.8h       \n"                        \
  "fmax   v18.8h, v18.8h, v0.8h       \n"                        \
  "fmin   v4.8h,  v4.8h, v3.8h        \n"                        \
  "fmin   v6.8h,  v6.8h, v3.8h        \n"                        \
  "fmin   v8.8h,  v8.8h, v3.8h        \n"                        \
  "fmin   v10.8h, v10.8h, v3.8h       \n"                        \
  "fmin   v12.8h, v12.8h, v3.8h       \n"                        \
  "fmin   v14.8h, v14.8h, v3.8h       \n"                        \
  "fmin   v16.8h, v16.8h, v3.8h       \n"                        \
  "fmin   v18.8h, v18.8h, v3.8h       \n"                        \
  "fmul   v20.8h,  v5.8h,  v4.8h        \n"                      \
  "fmul   v21.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v22.8h,  v9.8h,  v8.8h        \n"                      \
  "fmul   v23.8h,  v11.8h, v10.8h        \n"                     \
  "fmul   v24.8h,  v13.8h, v12.8h        \n"                     \
  "fmul   v25.8h,  v15.8h, v14.8h        \n"                     \
  "fmul   v26.8h,  v17.8h, v16.8h        \n"                     \
  "fmul   v27.8h,  v19.8h,  v18.8h        \n"                    \
  "fadd   v4.8h,  v28.8h, v1.8h        \n"                       \
  "fadd   v6.8h,  v29.8h, v1.8h        \n"                       \
  "fadd   v8.8h,  v30.8h, v1.8h        \n"                       \
  "fadd   v10.8h,  v31.8h, v1.8h        \n"                      \
  "fmul   v5.8h,   v28.8h, v2.8h        \n"                      \
  "fmul   v7.8h,   v29.8h, v2.8h        \n"                      \
  "fmul   v9.8h,   v30.8h, v2.8h        \n"                      \
  "fmul   v11.8h,  v31.8h, v2.8h        \n"                      \
  "fmax   v4.8h,  v4.8h, v0.8h        \n"                        \
  "fmax   v6.8h,  v6.8h, v0.8h        \n"                        \
  "fmax   v8.8h,  v8.8h, v0.8h        \n"                        \
  "fmax   v10.8h, v10.8h, v0.8h       \n"                        \
  "fmin   v4.8h,  v4.8h, v3.8h        \n"                        \
  "fmin   v6.8h,  v6.8h, v3.8h        \n"                        \
  "fmin   v8.8h,  v8.8h, v3.8h        \n"                        \
  "fmin   v10.8h, v10.8h, v3.8h       \n"                        \
  "fmul   v28.8h,  v5.8h,  v4.8h        \n"                      \
  "fmul   v29.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v30.8h,  v9.8h,  v8.8h        \n"                      \
  "fmul   v31.8h,  v11.8h, v10.8h        \n"                     \
  "9:\n"

#define SPARSE_F16_F16_W48_SEMI1_V8_HARD_SWISH                   \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v6.8h,  v20.8h, v1.8h        \n"                       \
  "fadd   v8.8h,  v21.8h, v1.8h        \n"                       \
  "fadd   v10.8h,  v22.8h, v1.8h        \n"                      \
  "fadd   v12.8h,  v23.8h, v1.8h        \n"                      \
  "fadd   v14.8h,  v24.8h, v1.8h        \n"                      \
  "fadd   v16.8h,  v25.8h, v1.8h        \n"                      \
  "fmul   v7.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v9.8h,   v21.8h, v2.8h        \n"                      \
  "fmul   v11.8h,  v22.8h, v2.8h        \n"                      \
  "fmul   v13.8h,  v23.8h, v2.8h        \n"                      \
  "fmul   v15.8h,  v24.8h, v2.8h        \n"                      \
  "fmul   v17.8h,  v25.8h, v2.8h        \n"                      \
  "fmax   v6.8h,  v6.8h, v0.8h        \n"                        \
  "fmax   v8.8h,  v8.8h, v0.8h        \n"                        \
  "fmax   v10.8h, v10.8h, v0.8h       \n"                        \
  "fmax   v12.8h, v12.8h, v0.8h       \n"                        \
  "fmax   v14.8h, v14.8h, v0.8h       \n"                        \
  "fmax   v16.8h, v16.8h, v0.8h       \n"                        \
  "fmin   v6.8h,  v6.8h, v3.8h        \n"                        \
  "fmin   v8.8h,  v8.8h, v3.8h        \n"                        \
  "fmin   v10.8h, v10.8h, v3.8h       \n"                        \
  "fmin   v12.8h, v12.8h, v3.8h       \n"                        \
  "fmin   v14.8h, v14.8h, v3.8h       \n"                        \
  "fmin   v16.8h, v16.8h, v3.8h       \n"                        \
  "fmul   v20.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v21.8h,  v9.8h,  v8.8h        \n"                      \
  "fmul   v22.8h,  v11.8h, v10.8h        \n"                     \
  "fmul   v23.8h,  v13.8h, v12.8h        \n"                     \
  "fmul   v24.8h,  v15.8h, v14.8h        \n"                     \
  "fmul   v25.8h,  v17.8h, v16.8h        \n"                     \
  "9:\n"

#define SPARSE_F16_F16_W32_SEMI2_V8_HARD_SWISH                   \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v4.8h,   v20.8h, v1.8h        \n"                      \
  "fadd   v6.8h,   v21.8h, v1.8h        \n"                      \
  "fadd   v8.8h,   v22.8h, v1.8h        \n"                      \
  "fadd   v10.8h,  v23.8h, v1.8h        \n"                      \
  "fadd   v12.8h,  v26.8h, v1.8h        \n"                      \
  "fadd   v14.8h,  v27.8h, v1.8h        \n"                      \
  "fadd   v16.8h,  v28.8h, v1.8h        \n"                      \
  "fadd   v18.8h,  v29.8h, v1.8h        \n"                      \
  "fmul   v5.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v7.8h,   v21.8h, v2.8h        \n"                      \
  "fmul   v9.8h,   v22.8h, v2.8h        \n"                      \
  "fmul   v11.8h,  v23.8h, v2.8h        \n"                      \
  "fmul   v13.8h,  v26.8h, v2.8h        \n"                      \
  "fmul   v15.8h,  v27.8h, v2.8h        \n"                      \
  "fmul   v17.8h,  v28.8h, v2.8h        \n"                      \
  "fmul   v19.8h,  v29.8h, v2.8h        \n"                      \
  "fmax   v4.8h,   v4.8h, v0.8h        \n"                       \
  "fmax   v6.8h,   v6.8h, v0.8h        \n"                       \
  "fmax   v8.8h,   v8.8h, v0.8h        \n"                       \
  "fmax   v10.8h,  v10.8h, v0.8h       \n"                       \
  "fmax   v12.8h,  v12.8h, v0.8h       \n"                       \
  "fmax   v14.8h,  v14.8h, v0.8h       \n"                       \
  "fmax   v16.8h,  v16.8h, v0.8h       \n"                       \
  "fmax   v18.8h,  v18.8h, v0.8h       \n"                       \
  "fmin   v4.8h,   v4.8h, v3.8h        \n"                       \
  "fmin   v6.8h,   v6.8h, v3.8h        \n"                       \
  "fmin   v8.8h,   v8.8h, v3.8h        \n"                       \
  "fmin   v10.8h,  v10.8h, v3.8h       \n"                       \
  "fmin   v12.8h,  v12.8h, v3.8h       \n"                       \
  "fmin   v14.8h,  v14.8h, v3.8h       \n"                       \
  "fmin   v16.8h,  v16.8h, v3.8h       \n"                       \
  "fmin   v18.8h,  v18.8h, v3.8h       \n"                       \
  "fmul   v20.8h,  v5.8h,  v4.8h        \n"                      \
  "fmul   v21.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v22.8h,  v9.8h,  v8.8h        \n"                      \
  "fmul   v23.8h,  v11.8h, v10.8h        \n"                     \
  "fmul   v26.8h,  v13.8h, v12.8h        \n"                     \
  "fmul   v27.8h,  v15.8h, v14.8h        \n"                     \
  "fmul   v28.8h,  v17.8h, v16.8h        \n"                     \
  "fmul   v29.8h,  v19.8h,  v18.8h        \n"                    \
  "9:\n"

#define SPARSE_F16_F16_W32_SEMI1_V8_HARD_SWISH                   \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v6.8h,  v20.8h, v1.8h        \n"                       \
  "fadd   v8.8h,  v21.8h, v1.8h        \n"                       \
  "fadd   v10.8h,  v22.8h, v1.8h        \n"                      \
  "fadd   v12.8h,  v23.8h, v1.8h        \n"                      \
  "fmul   v7.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v9.8h,   v21.8h, v2.8h        \n"                      \
  "fmul   v11.8h,  v22.8h, v2.8h        \n"                      \
  "fmul   v13.8h,  v23.8h, v2.8h        \n"                      \
  "fmax   v6.8h,  v6.8h, v0.8h        \n"                        \
  "fmax   v8.8h,  v8.8h, v0.8h        \n"                        \
  "fmax   v10.8h, v10.8h, v0.8h       \n"                        \
  "fmax   v12.8h, v12.8h, v0.8h       \n"                        \
  "fmin   v6.8h,  v6.8h, v3.8h        \n"                        \
  "fmin   v8.8h,  v8.8h, v3.8h        \n"                        \
  "fmin   v10.8h, v10.8h, v3.8h       \n"                        \
  "fmin   v12.8h, v12.8h, v3.8h       \n"                        \
  "fmul   v20.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v21.8h,  v9.8h,  v8.8h        \n"                      \
  "fmul   v22.8h,  v11.8h, v10.8h        \n"                     \
  "fmul   v23.8h,  v13.8h, v12.8h        \n"                     \
  "9:\n"

#define SPARSE_F16_F16_W16_SEMI2_V8_HARD_SWISH                   \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v4.8h,   v20.8h, v1.8h        \n"                      \
  "fadd   v6.8h,   v21.8h, v1.8h        \n"                      \
  "fadd   v12.8h,  v26.8h, v1.8h        \n"                      \
  "fadd   v14.8h,  v27.8h, v1.8h        \n"                      \
  "fmul   v5.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v7.8h,   v21.8h, v2.8h        \n"                      \
  "fmul   v13.8h,  v26.8h, v2.8h        \n"                      \
  "fmul   v15.8h,  v27.8h, v2.8h        \n"                      \
  "fmax   v4.8h,   v4.8h, v0.8h        \n"                       \
  "fmax   v6.8h,   v6.8h, v0.8h        \n"                       \
  "fmax   v12.8h,  v12.8h, v0.8h       \n"                       \
  "fmax   v14.8h,  v14.8h, v0.8h       \n"                       \
  "fmin   v4.8h,   v4.8h, v3.8h        \n"                       \
  "fmin   v6.8h,   v6.8h, v3.8h        \n"                       \
  "fmin   v12.8h,  v12.8h, v3.8h       \n"                       \
  "fmin   v14.8h,  v14.8h, v3.8h       \n"                       \
  "fmul   v20.8h,  v5.8h,  v4.8h        \n"                      \
  "fmul   v21.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v26.8h,  v13.8h, v12.8h        \n"                     \
  "fmul   v27.8h,  v15.8h, v14.8h        \n"                     \
  "9:\n"

#define SPARSE_F16_F16_W16_SEMI1_V8_HARD_SWISH                   \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v6.8h,  v20.8h, v1.8h        \n"                       \
  "fadd   v8.8h,  v21.8h, v1.8h        \n"                       \
  "fmul   v7.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v9.8h,   v21.8h, v2.8h        \n"                      \
  "fmax   v6.8h,  v6.8h, v0.8h        \n"                        \
  "fmax   v8.8h,  v8.8h, v0.8h        \n"                        \
  "fmin   v6.8h,  v6.8h, v3.8h        \n"                        \
  "fmin   v8.8h,  v8.8h, v3.8h        \n"                        \
  "fmul   v20.8h,  v7.8h,  v6.8h        \n"                      \
  "fmul   v21.8h,  v9.8h,  v8.8h        \n"                      \
  "9:\n"

#define SPARSE_F16_F16_W8_SEMI2_V8_HARD_SWISH                    \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v4.8h,   v20.8h, v1.8h        \n"                      \
  "fadd   v12.8h,  v26.8h, v1.8h        \n"                      \
  "fmul   v5.8h,   v20.8h, v2.8h        \n"                      \
  "fmul   v13.8h,  v26.8h, v2.8h        \n"                      \
  "fmax   v4.8h,   v4.8h, v0.8h        \n"                       \
  "fmax   v12.8h,  v12.8h, v0.8h       \n"                       \
  "fmin   v4.8h,   v4.8h, v3.8h        \n"                       \
  "fmin   v12.8h,  v12.8h, v3.8h       \n"                       \
  "fmul   v20.8h,  v5.8h,  v4.8h        \n"                      \
  "fmul   v26.8h,  v13.8h, v12.8h        \n"                     \
  "9:\n"

#define SPARSE_F16_F16_W8_SEMI1_V8_HARD_SWISH                    \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v6.8h,  v20.8h, v1.8h        \n"                       \
  "fmul   v7.8h,   v20.8h, v2.8h        \n"                      \
  "fmax   v6.8h,  v6.8h, v0.8h        \n"                        \
  "fmin   v6.8h,  v6.8h, v3.8h        \n"                        \
  "fmul   v20.8h,  v7.8h,  v6.8h        \n"                      \
  "9:\n"

#define SPARSE_F16_F16_W4_SEMI2_V8_HARD_SWISH                    \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v4.4h,   v20.4h, v1.4h        \n"                      \
  "fadd   v12.4h,  v26.4h, v1.4h        \n"                      \
  "fmul   v5.4h,   v20.4h, v2.4h        \n"                      \
  "fmul   v13.4h,  v26.4h, v2.4h        \n"                      \
  "fmax   v4.4h,   v4.4h,  v0.4h        \n"                      \
  "fmax   v12.4h,  v12.4h, v0.4h       \n"                       \
  "fmin   v4.4h,   v4.4h,  v3.4h        \n"                      \
  "fmin   v12.4h,  v12.4h, v3.4h       \n"                       \
  "fmul   v20.4h,  v5.4h,  v4.4h        \n"                      \
  "fmul   v26.4h,  v13.4h, v12.4h        \n"                     \
  "9:\n"

#define SPARSE_F16_F16_W4_SEMI1_V8_HARD_SWISH                    \
  /* do hard_swish */                                            \
  "12: \n"                                                       \
  "movi   v0.8h,    #0                \n"    /* for hardswish */ \
  "ldr    q1,  [%[hs_param], #0]         \n" /* offset */        \
  "ldr    q2,  [%[hs_param], #16]        \n" /* scale */         \
  "ldr    q3,  [%[hs_param], #32]        \n" /* threshold */     \
  "fadd   v6.4h,  v20.4h, v1.4h        \n"                       \
  "fmul   v7.4h,   v20.4h, v2.4h        \n"                      \
  "fmax   v6.4h,  v6.4h, v0.4h        \n"                        \
  "fmin   v6.4h,  v6.4h, v3.4h        \n"                        \
  "fmul   v20.4h,  v7.4h,  v6.4h        \n"                      \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_F16_F16_W48_SEMI2_V8_OUT  \
  SPARSE_F16_F16_W48_SEMI2_V8_KERNEL     \
  SPARSE_F16_F16_W48_SEMI2_V8_RELU       \
  SPARSE_F16_F16_W48_SEMI2_V8_RELU6      \
  SPARSE_F16_F16_W48_SEMI2_V8_LEAKY_RELU \
  SPARSE_F16_F16_W48_SEMI2_V8_HARD_SWISH \
  /* store result */                     \
  "stp   q20, q21,  [%[c_ptr1]]\n"       \
  "stp   q26, q27,  [%[c_ptr2]]\n"       \
  "stp   q22, q23,  [%[c_ptr1], #32]\n"  \
  "stp   q28, q29,  [%[c_ptr2], #32]\n"  \
  "stp   q24, q25,  [%[c_ptr1], #64]\n"  \
  "stp   q30, q31,  [%[c_ptr2], #64]\n"

#define SPARSE_F16_F16_W48_SEMI1_V8_OUT  \
  SPARSE_F16_F16_W48_SEMI1_V8_KERNEL     \
  SPARSE_F16_F16_W48_SEMI1_V8_RELU       \
  SPARSE_F16_F16_W48_SEMI1_V8_RELU6      \
  SPARSE_F16_F16_W48_SEMI1_V8_LEAKY_RELU \
  SPARSE_F16_F16_W48_SEMI1_V8_HARD_SWISH \
  /* store result */                     \
  "stp   q20, q21,  [%[c_ptr]]\n"        \
  "stp   q22, q23,  [%[c_ptr], #32]\n"   \
  "stp   q24, q25,  [%[c_ptr], #64]\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx32.
 */
#define SPARSE_F16_F16_W32_SEMI2_V8_OUT  \
  SPARSE_F16_F16_W32_SEMI2_V8_KERNEL     \
  SPARSE_F16_F16_W32_SEMI2_V8_RELU       \
  SPARSE_F16_F16_W32_SEMI2_V8_RELU6      \
  SPARSE_F16_F16_W32_SEMI2_V8_LEAKY_RELU \
  SPARSE_F16_F16_W32_SEMI2_V8_HARD_SWISH \
  /* store result */                     \
  "stp   q20, q21,  [%[c_ptr1]]\n"       \
  "stp   q26, q27,  [%[c_ptr2]]\n"       \
  "stp   q22, q23,  [%[c_ptr1], #32]\n"  \
  "stp   q28, q29,  [%[c_ptr2], #32]\n"

#define SPARSE_F16_F16_W32_SEMI1_V8_OUT  \
  SPARSE_F16_F16_W32_SEMI1_V8_KERNEL     \
  SPARSE_F16_F16_W32_SEMI1_V8_RELU       \
  SPARSE_F16_F16_W32_SEMI1_V8_RELU6      \
  SPARSE_F16_F16_W32_SEMI1_V8_LEAKY_RELU \
  SPARSE_F16_F16_W32_SEMI1_V8_HARD_SWISH \
  /* store result */                     \
  "stp   q20, q21,  [%[c_ptr]]\n"        \
  "stp   q22, q23,  [%[c_ptr], #32]\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_F16_F16_W16_SEMI2_V8_OUT  \
  SPARSE_F16_F16_W16_SEMI2_V8_KERNEL     \
  SPARSE_F16_F16_W16_SEMI2_V8_RELU       \
  SPARSE_F16_F16_W16_SEMI2_V8_RELU6      \
  SPARSE_F16_F16_W16_SEMI2_V8_LEAKY_RELU \
  SPARSE_F16_F16_W16_SEMI2_V8_HARD_SWISH \
  /* store result */                     \
  "stp   q20, q21,  [%[c_ptr1]]\n"       \
  "stp   q26, q27,  [%[c_ptr2]]\n"

#define SPARSE_F16_F16_W16_SEMI1_V8_OUT  \
  SPARSE_F16_F16_W16_SEMI1_V8_KERNEL     \
  SPARSE_F16_F16_W16_SEMI1_V8_RELU       \
  SPARSE_F16_F16_W16_SEMI1_V8_RELU6      \
  SPARSE_F16_F16_W16_SEMI1_V8_LEAKY_RELU \
  SPARSE_F16_F16_W16_SEMI1_V8_HARD_SWISH \
  /* store result */                     \
  "stp   q20, q21,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F16_F16_W8_SEMI2_V8_OUT  \
  SPARSE_F16_F16_W8_SEMI2_V8_KERNEL     \
  SPARSE_F16_F16_W8_SEMI2_V8_RELU       \
  SPARSE_F16_F16_W8_SEMI2_V8_RELU6      \
  SPARSE_F16_F16_W8_SEMI2_V8_LEAKY_RELU \
  SPARSE_F16_F16_W8_SEMI2_V8_HARD_SWISH \
  /* store result */                    \
  "str   q20,  [%[c_ptr1]]\n"           \
  "str   q26,  [%[c_ptr2]]\n"

#define SPARSE_F16_F16_W8_SEMI1_V8_OUT  \
  SPARSE_F16_F16_W8_SEMI1_V8_KERNEL     \
  SPARSE_F16_F16_W8_SEMI1_V8_RELU       \
  SPARSE_F16_F16_W8_SEMI1_V8_RELU6      \
  SPARSE_F16_F16_W8_SEMI1_V8_LEAKY_RELU \
  SPARSE_F16_F16_W8_SEMI1_V8_HARD_SWISH \
  /* store result */                    \
  "str   q20,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F16_F16_W4_SEMI2_V8_OUT  \
  SPARSE_F16_F16_W4_SEMI2_V8_KERNEL     \
  SPARSE_F16_F16_W4_SEMI2_V8_RELU       \
  SPARSE_F16_F16_W4_SEMI2_V8_RELU6      \
  SPARSE_F16_F16_W4_SEMI2_V8_LEAKY_RELU \
  SPARSE_F16_F16_W4_SEMI2_V8_HARD_SWISH \
  /* store result */                    \
  "str   d20,  [%[c_ptr1]]\n"           \
  "str   d26,  [%[c_ptr2]]\n"

#define SPARSE_F16_F16_W4_SEMI1_V8_OUT  \
  SPARSE_F16_F16_W4_SEMI1_V8_KERNEL     \
  SPARSE_F16_F16_W4_SEMI1_V8_RELU       \
  SPARSE_F16_F16_W4_SEMI1_V8_RELU6      \
  SPARSE_F16_F16_W4_SEMI1_V8_LEAKY_RELU \
  SPARSE_F16_F16_W4_SEMI1_V8_HARD_SWISH \
  /* store result */                    \
  "str   d20,  [%[c_ptr]]\n"

#define SET_ASSM_INPUT_PARAM1_V8_F16                               \
  : [a_ptr] "+r"(cur_w),                  \
    [b_ptr] "+r"(cur_b),                  \
    [c_ptr1] "+r"(out_ptr1),              \
    [c_ptr2] "+r"(out_ptr2),              \
    [k] "+r"(nnz),                        \
    [widx_dmap] "+r"(dmap),               \
    [bias_ptr] "+r"(pbias)                \
  : [vflag_act] "r"(flag_act),            \
    [valpha] "r"(alpha),                  \
    [hs_param] "r"(hs_param)              \
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", \
    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", \
    "v16", "v17", "v18", "v19", "v21", "v22", "v23", "v24", "v25",  \
    "v26", "v27", "v28", "v29", "v30", "v31", "w1", "x1", "cc", "memory"

#define SET_ASSM_INPUT_PARAM2_V8_F16                               \
  : [a_ptr] "+r"(cur_w),                  \
    [b_ptr] "+r"(cur_b),                  \
    [c_ptr] "+r"(out_ptr),                \
    [k] "+r"(nnz),                        \
    [widx_dmap] "+r"(dmap)                \
  : [vbias] "r"(vbias),                   \
    [vflag_act] "r"(flag_act),            \
    [valpha] "r"(alpha),                  \
    [hs_param] "r"(hs_param) \
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", \
    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", \
    "v16", "v17", "v18", "v19", "v21", "v22", "v23", "v24", "v25",  \
    "v26", "v27", "v28", "v29", "v30", "v31", "w1", "x1", "cc", "memory"

/**
 * \brief Semi-structured Sparse calculation implementation of 1x1 convolution,
 * both input and output are f16.
 * Semi-structured Sparse matrix multiplication is calculated in blocks, the
 * block size is Mx48,
 * that is,
 * the parameter matrix is MxK, and the activation matrix is Kx48; when N is
 * less than 48,
 * it is calculated in blocks of Mx32, Mx16, Mx8, and Mx4 in turn;
 * @param A Semi-structured Sparse weight data
 * @param B dense input data
 * @param widx_dmap An array of int32_t values storing scaled [by sizeof(input
 * element)] difference
 * between input channels corresponding to successive non-zero element
 * @param nidx_nnzmap the number of non-zero kernel elements per each output
 * channel
 * @param bias
 * @param output
 * @param M
 * @param N
 * @param K
 * @param param
 * @param ctx
 */
void sparse_semi_conv_fp16_pipelined(const float16_t* A,
                                     const float16_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float16_t* bias,
                                     float16_t* output,
                                     const int M,
                                     const int K,
                                     const int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  INIT_SEMI_CONV_PARAM(float16_t, float16_t, 48)
  float16_t bias_zero[2] = {0.f, 0.f};
  while
    SPARSE_FP16_LIKELY(mc >= 48 * sizeof(float16_t)) {
      LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
        GET_SEMI_PARAM_TABLE(float16_t, float16_t)
        // clang-format off
            asm volatile(SPARSE_F16_F16_W48_SEMI2_V8_OUT  
              SET_ASSM_INPUT_PARAM1_V8_F16);
        // clang-format on
      }
      LITE_PARALLEL_COMMON_END();
      if
        SPARSE_FP16_UNLIKELY(lave_num != 0) {
          GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
          // clang-format off
          asm volatile(SPARSE_F16_F16_W48_SEMI1_V8_OUT  
            SET_ASSM_INPUT_PARAM2_V8_F16);
          // clang-format on
        }
      output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                            48 * sizeof(float16_t));
      B += 48;
      mc -= 48 * sizeof(float16_t);
    }

  if
    SPARSE_FP16_UNLIKELY(mc != 0) {
      if (mc >= (32 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          // clang-format off
              asm volatile(SPARSE_F16_F16_W32_SEMI2_V8_OUT  
                SET_ASSM_INPUT_PARAM1_V8_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            // clang-format off
            asm volatile(SPARSE_F16_F16_W32_SEMI1_V8_OUT  
              SET_ASSM_INPUT_PARAM2_V8_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              32 * sizeof(float16_t));
        B += 32;
        mc -= 32 * sizeof(float16_t);
      }
      if (mc >= (16 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          // clang-format off
              asm volatile(SPARSE_F16_F16_W16_SEMI2_V8_OUT  
                SET_ASSM_INPUT_PARAM1_V8_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            // clang-format off
            asm volatile(SPARSE_F16_F16_W16_SEMI1_V8_OUT  
              SET_ASSM_INPUT_PARAM2_V8_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              16 * sizeof(float16_t));
        B += 16;
        mc -= 16 * sizeof(float16_t);
      }
      if (mc >= (8 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          // clang-format off
              asm volatile(SPARSE_F16_F16_W8_SEMI2_V8_OUT  
                SET_ASSM_INPUT_PARAM1_V8_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            // clang-format off
            asm volatile(SPARSE_F16_F16_W8_SEMI1_V8_OUT  
              SET_ASSM_INPUT_PARAM2_V8_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              8 * sizeof(float16_t));
        B += 8;
        mc -= 8 * sizeof(float16_t);
      }
      if (mc >= (4 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          // clang-format off
              asm volatile(SPARSE_F16_F16_W4_SEMI2_V8_OUT  
                SET_ASSM_INPUT_PARAM1_V8_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            // clang-format off
            asm volatile(SPARSE_F16_F16_W4_SEMI1_V8_OUT  
              SET_ASSM_INPUT_PARAM2_V8_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              4 * sizeof(float16_t));
        B += 4;
        mc -= 4 * sizeof(float16_t);
      }
      while (mc >= (1 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          float16_t vacc01n0 = pbias[0];
          float16_t vacc01n1 = pbias[1];
          if
            SPARSE_FP16_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                vacc01n0 += cur_b[0] * cur_w[0];
                vacc01n1 += cur_b[0] * cur_w[1];
                cur_w += 2;
                cur_b = (const float16_t*)((uintptr_t)cur_b + (uintptr_t)diff);
              } while (--nnz != 0);
            }

          COMPUTE_ACT_NEON_TWO_V8_F32
          out_ptr1[0] = vacc01n0;
          out_ptr2[0] = vacc01n1;
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            float16_t vacc01n0 = vbias;
            if
              SPARSE_FP16_LIKELY(nnz != 0) {
                do {
                  const intptr_t diff = *dmap++;
                  vacc01n0 += cur_b[0] * cur_w[0];
                  cur_b =
                      (const float16_t*)((uintptr_t)cur_b + (uintptr_t)diff);
                  cur_w += 1;
                } while (--nnz != 0);
              }
            COMPUTE_ACT_NEON_ONE_V8_F32
            out_ptr[0] = vacc01n0;
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              1 * sizeof(float16_t));
        B += 1;
        mc -= 1 * sizeof(float16_t);
      }
    }
}

#else  // armv7

#define SPARSE_F16_F16_W32_SEMI2_v7_KERNEL \
  "vld1.16    {d0},   [%[bias_ptr]]\n"     \
  "pld  [%[a_ptr], #64]    \n"             \
  "pld  [%[widx_dmap], #64]    \n"         \
  "vdup.16    q8,    d0[0]\n"              \
  "vdup.16    q9,    d0[0]\n"              \
  "vdup.16    q10,   d0[0]\n"              \
  "vdup.16    q11,   d0[0]\n"              \
  "vdup.16    q12,   d0[1]\n"              \
  "pld  [%[b_ptr], #192]    \n"            \
  "vdup.16    q13,   d0[1]\n"              \
  "vdup.16    q14,   d0[1]\n"              \
  "vdup.16    q15,   d0[1]\n"              \
  "cmp    %[k], #0\n"                      \
  "beq 1f\n"                               \
  "0:\n"                                   \
  "vld1.16  d0, [%[a_ptr]]\n"              \
  "ldr   r0, [%[widx_dmap]],   #4\n"       \
  "mov   r2,   %[b_ptr]\n"                 \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"         \
  "add r2, r2, #32\n"                      \
  "add %[a_ptr], %[a_ptr], #4\n"           \
  "vld1.16  {d6-d9}, [r2]\n"               \
  "vmla.f16    q8,   q1,  d0[0]\n"         \
  "vmla.f16    q12,  q1,  d0[1]\n"         \
  "vmla.f16    q9,   q2,  d0[0]\n"         \
  "vmla.f16    q13,  q2,  d0[1]\n"         \
  "add   %[b_ptr],  %[b_ptr], r0\n"        \
  "subs    %[k],   %[k],   #1\n"           \
  "vmla.f16    q10,  q3,  d0[0]\n"         \
  "vmla.f16    q14,  q3,  d0[1]\n"         \
  "vmla.f16    q11,  q4,  d0[0]\n"         \
  "vmla.f16    q15,  q4,  d0[1]\n"         \
  "pld  [%[widx_dmap], #128]    \n"        \
  "bne     0b\n"                           \
  "1:\n"

#define SPARSE_F16_F16_W32_SEMI1_v7_KERNEL \
  "vdup.16    q0,    %[vbias]\n"           \
  "pld  [%[a_ptr], #64]    \n"             \
  "pld  [%[widx_dmap], #64]    \n"         \
  "vdup.16    q8,   d0[0]\n"               \
  "vdup.16    q9,   d0[0]\n"               \
  "pld  [%[b_ptr], #192]    \n"            \
  "vdup.16    q10,   d0[0]\n"              \
  "vdup.16    q11,   d0[0]\n"              \
  "cmp    %[k], #0\n"                      \
  "beq 1f\n"                               \
  "0:\n"                                   \
  "vld1.16  d0, [%[a_ptr]]\n"              \
  "mov   r2,   %[b_ptr]\n"                 \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"         \
  "add r2, r2, #32\n"                      \
  "add %[a_ptr], %[a_ptr], #2\n"           \
  "vld1.16  {d6-d9}, [r2]\n"               \
  "pld  [%[widx_dmap], #128]    \n"        \
  "ldr   r0, [%[widx_dmap]],   #4\n"       \
  "subs    %[k],   %[k],   #1\n"           \
  "vmla.f16    q8,    q1,  d0[0]\n"        \
  "vmla.f16    q9,    q2,  d0[0]\n"        \
  "add   %[b_ptr],  %[b_ptr], r0\n"        \
  "vmla.f16    q10,   q3,  d0[0]\n"        \
  "vmla.f16    q11,   q4,  d0[0]\n"        \
  "bne     0b\n"                           \
  "1:\n"

#define SPARSE_F16_F16_W16_SEMI2_v7_KERNEL \
  "vld1.16    {d0},   [%[bias_ptr]]\n"     \
  "pld  [%[a_ptr], #64]    \n"             \
  "pld  [%[widx_dmap], #64]    \n"         \
  "vdup.16    q8,    d0[0]\n"              \
  "vdup.16    q9,    d0[0]\n"              \
  "vdup.16    q12,   d0[1]\n"              \
  "pld  [%[b_ptr], #192]    \n"            \
  "vdup.16    q13,   d0[1]\n"              \
  "cmp    %[k], #0\n"                      \
  "beq 1f\n"                               \
  "0:\n"                                   \
  "vld1.16  d0, [%[a_ptr]]\n"              \
  "ldr   r0, [%[widx_dmap]],   #4\n"       \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"         \
  "add %[a_ptr], %[a_ptr], #4\n"           \
  "add   %[b_ptr],  %[b_ptr], r0\n"        \
  "subs    %[k],   %[k],   #1\n"           \
  "vmla.f16    q8,   q1,  d0[0]\n"         \
  "vmla.f16    q12,  q1,  d0[1]\n"         \
  "vmla.f16    q9,   q2,  d0[0]\n"         \
  "vmla.f16    q13,  q2,  d0[1]\n"         \
  "pld  [%[widx_dmap], #128]    \n"        \
  "bne     0b\n"                           \
  "1:\n"

#define SPARSE_F16_F16_W16_SEMI1_v7_KERNEL \
  "vdup.16    q0,    %[vbias]\n"           \
  "pld  [%[a_ptr], #64]    \n"             \
  "pld  [%[widx_dmap], #64]    \n"         \
  "vdup.16    q8,   d0[0]\n"               \
  "vdup.16    q9,   d0[0]\n"               \
  "pld  [%[b_ptr], #192]    \n"            \
  "cmp    %[k], #0\n"                      \
  "beq 1f\n"                               \
  "0:\n"                                   \
  "vld1.16  d0, [%[a_ptr]]\n"              \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"         \
  "add %[a_ptr], %[a_ptr], #2\n"           \
  "pld  [%[widx_dmap], #128]    \n"        \
  "ldr   r0, [%[widx_dmap]],   #4\n"       \
  "subs    %[k],   %[k],   #1\n"           \
  "add   %[b_ptr],  %[b_ptr], r0\n"        \
  "vmla.f16    q8,    q1,  d0[0]\n"        \
  "vmla.f16    q9,    q2,  d0[0]\n"        \
  "bne     0b\n"                           \
  "1:\n"

#define SPARSE_F16_F16_W8_SEMI2_v7_KERNEL \
  "vld1.16    {d0},   [%[bias_ptr]]\n"    \
  "pld  [%[a_ptr], #64]    \n"            \
  "pld  [%[widx_dmap], #64]    \n"        \
  "vdup.16    q8,    d0[0]\n"             \
  "vdup.16    q12,   d0[1]\n"             \
  "pld  [%[b_ptr], #192]    \n"           \
  "cmp    %[k], #0\n"                     \
  "beq 1f\n"                              \
  "0:\n"                                  \
  "vld1.16  d0, [%[a_ptr]]\n"             \
  "ldr   r0, [%[widx_dmap]],   #4\n"      \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"        \
  "add %[a_ptr], %[a_ptr], #4\n"          \
  "add   %[b_ptr],  %[b_ptr], r0\n"       \
  "subs    %[k],   %[k],   #1\n"          \
  "vmla.f16    q8,   q1,  d0[0]\n"        \
  "vmla.f16    q12,  q1,  d0[1]\n"        \
  "pld  [%[widx_dmap], #128]    \n"       \
  "bne     0b\n"                          \
  "1:\n"

#define SPARSE_F16_F16_W8_SEMI1_v7_KERNEL \
  "vdup.16    q0,    %[vbias]\n"          \
  "pld  [%[a_ptr], #64]    \n"            \
  "pld  [%[widx_dmap], #64]    \n"        \
  "vdup.16    q8,   d0[0]\n"              \
  "pld  [%[b_ptr], #192]    \n"           \
  "cmp    %[k], #0\n"                     \
  "beq 1f\n"                              \
  "0:\n"                                  \
  "vld1.16  d0, [%[a_ptr]]\n"             \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"        \
  "add %[a_ptr], %[a_ptr], #2\n"          \
  "pld  [%[widx_dmap], #128]    \n"       \
  "ldr   r0, [%[widx_dmap]],   #4\n"      \
  "subs    %[k],   %[k],   #1\n"          \
  "add   %[b_ptr],  %[b_ptr], r0\n"       \
  "vmla.f16    q8,    q1,  d0[0]\n"       \
  "bne     0b\n"                          \
  "1:\n"

#define SPARSE_F16_F16_W4_SEMI2_v7_KERNEL \
  "vld1.16    {d0},   [%[bias_ptr]]\n"    \
  "pld  [%[a_ptr], #64]    \n"            \
  "pld  [%[widx_dmap], #64]    \n"        \
  "vdup.16    q8,    d0[0]\n"             \
  "vdup.16    q12,   d0[1]\n"             \
  "pld  [%[b_ptr], #192]    \n"           \
  "cmp    %[k], #0\n"                     \
  "beq 1f\n"                              \
  "0:\n"                                  \
  "vld1.16  d0, [%[a_ptr]]\n"             \
  "ldr   r0, [%[widx_dmap]],   #4\n"      \
  "vld1.16  {d2}, [%[b_ptr]]\n"           \
  "add %[a_ptr], %[a_ptr], #4\n"          \
  "add   %[b_ptr],  %[b_ptr], r0\n"       \
  "subs    %[k],   %[k],   #1\n"          \
  "vmla.f16    d16,  d2,  d0[0]\n"        \
  "vmla.f16    d24,  d2,  d0[1]\n"        \
  "pld  [%[widx_dmap], #128]    \n"       \
  "bne     0b\n"                          \
  "1:\n"

#define SPARSE_F16_F16_W4_SEMI1_v7_KERNEL \
  "vdup.16    q0,    %[vbias]\n"          \
  "pld  [%[a_ptr], #64]    \n"            \
  "pld  [%[widx_dmap], #64]    \n"        \
  "vdup.16    q8,   d0[0]\n"              \
  "pld  [%[b_ptr], #192]    \n"           \
  "cmp    %[k], #0\n"                     \
  "beq 1f\n"                              \
  "0:\n"                                  \
  "vld1.16  d0, [%[a_ptr]]\n"             \
  "vld1.16  {d2}, [%[b_ptr]]\n"           \
  "add %[a_ptr], %[a_ptr], #2\n"          \
  "pld  [%[widx_dmap], #128]    \n"       \
  "ldr   r0, [%[widx_dmap]],   #4\n"      \
  "subs    %[k],   %[k],   #1\n"          \
  "add   %[b_ptr],  %[b_ptr], r0\n"       \
  "vmla.f16    d16,    d2,  d0[0]\n"      \
  "bne     0b\n"                          \
  "1:\n"

#define SPARSE_F16_F16_W32_SEMI2_v7_RELU         \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q8,   q8,   q0\n" /* relu */       \
  "vmax.f16   q9,   q9,   q0\n" /* relu */       \
  "vmax.f16   q10,  q10,  q0\n" /* relu */       \
  "vmax.f16   q11,  q11,  q0\n" /* relu */       \
  "vmax.f16   q12,  q12,  q0\n" /* relu */       \
  "vmax.f16   q13,  q13,  q0\n" /* relu */       \
  "vmax.f16   q14,  q14,  q0\n" /* relu */       \
  "vmax.f16   q15,  q15,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W32_SEMI1_v7_RELU         \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q10,  q10,  q0\n" /* relu */       \
  "vmax.f16   q11,  q11,  q0\n" /* relu */       \
  "vmax.f16   q12,  q12,  q0\n" /* relu */       \
  "vmax.f16   q13,  q13,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W16_SEMI2_v7_RELU         \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q8,   q8,   q0\n" /* relu */       \
  "vmax.f16   q9,   q9,   q0\n" /* relu */       \
  "vmax.f16   q12,  q12,  q0\n" /* relu */       \
  "vmax.f16   q13,  q13,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W16_SEMI1_v7_RELU         \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q8,  q8,  q0\n"   /* relu */       \
  "vmax.f16   q9,  q9,  q0\n"   /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W8_SEMI2_v7_RELU          \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q8,   q8,   q0\n" /* relu */       \
  "vmax.f16   q12,  q12,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W8_SEMI1_v7_RELU          \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q8,  q8,  q0\n"   /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W4_SEMI2_v7_RELU          \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   d16,  d16,  d0\n" /* relu */       \
  "vmax.f16   d24,  d24,  d0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W4_SEMI1_v7_RELU          \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   d0, #0\n"         /* for relu */   \
  "vmax.f16   d16,  d16,  d0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W32_SEMI2_v7_RELU6              \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f16   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f16   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f16   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f16   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f16   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f16   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f16   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f16   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f16   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f16   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f16   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f16   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W32_SEMI1_v7_RELU6              \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f16   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f16   q11,  q11,  q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f16   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f16   q11,  q11,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W32_SEMI2_v7_RELU6              \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f16   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f16   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f16   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f16   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f16   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f16   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f16   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f16   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f16   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f16   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f16   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f16   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W16_SEMI2_v7_RELU6              \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f16   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f16   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f16   q13,  q13,  q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f16   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f16   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f16   q13,  q13,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W16_SEMI1_v7_RELU6              \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f16   q9,   q9,   q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f16   q9,   q9,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W8_SEMI2_v7_RELU6               \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f16   q12,  q12,  q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f16   q12,  q12,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W8_SEMI1_v7_RELU6               \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q8,   q8,   q0\n"      /* relu6 */       \
  "vmin.f16   q8,   q8,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W4_SEMI2_v7_RELU6               \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   d16,  d16,  d0\n"      /* relu6 */       \
  "vmax.f16   d24,  d24,  d0\n"      /* relu6 */       \
  "vmin.f16   d16,  d16,   d2\n"     /* relu6 */       \
  "vmin.f16   d24,  d24,  d2\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W4_SEMI1_v7_RELU6               \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   d16,  d16,  d0\n"      /* relu6 */       \
  "vmin.f16   d16,  d16,   d2\n"     /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W32_SEMI2_v7_LEAKY_RELU                 \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q8,     q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q8,     q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q9,     q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q9,     q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q6,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q10,    q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                          \
  "vbif       q9,    q5,    q4    \n"                          \
  "vbif       q10,   q7,    q6    \n"                          \
  "vcge.f16   q2,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q11,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q12,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q6,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q11,    q3,    q2    \n"                         \
  "vbif       q12,    q5,    q4    \n"                         \
  "vbif       q13,    q7,    q6    \n"                         \
  "vcge.f16   q2,    q14,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q14,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q15,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q15,    q1    \n" /* vmulq_f32 */         \
  "vbif       q14,    q3,    q2    \n"                         \
  "vbif       q15,    q5,    q4    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI1_v7_LEAKY_RELU                 \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q8,    q0    \n"  /* vcgeq_f32 */         \
  "vmul.f16   q3,    q8,    q1    \n"  /* vmulq_f32 */         \
  "vcge.f16   q4,    q9,    q0    \n"  /* vcgeq_f32 */         \
  "vmul.f16   q5,    q9,    q1    \n"  /* vmulq_f32 */         \
  "vcge.f16   q6,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q10,    q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                          \
  "vcge.f16   q2,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q11,    q1    \n" /* vmulq_f32 */         \
  "vbif       q9,    q5,    q4    \n"                          \
  "vbif       q10,   q7,    q6    \n"                          \
  "vbif       q11,   q3,    q2    \n"                          \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI2_v7_LEAKY_RELU                 \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q8,     q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q8,     q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q9,     q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q9,     q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q6,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q12,    q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                          \
  "vbif       q9,    q5,    q4    \n"                          \
  "vbif       q12,   q7,    q6    \n"                          \
  "vcge.f16   q2,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q13,    q3,    q2    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_SEMI1_v7_LEAKY_RELU                \
  /* do relu */                                               \
  "11: \n"                                                    \
  "cmp   %[vflag_act],  #3       \n"  /* check leakey relu */ \
  "bne   12f                     \n"  /* no act end */        \
  "vmov.i16   q0, #0\n"               /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"       /* leakey relu alpha */ \
  "vcge.f16   q2,    q8,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q8,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q9,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q9,    q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                         \
  "vbif       q9,    q5,    q4    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI2_v7_LEAKY_RELU                  \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q8,     q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q8,     q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q6,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q12,    q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                          \
  "vbif       q12,   q7,    q6    \n"                          \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_SEMI1_v7_LEAKY_RELU                 \
  /* do relu */                                               \
  "11: \n"                                                    \
  "cmp   %[vflag_act],  #3       \n"  /* check leakey relu */ \
  "bne   12f                     \n"  /* no act end */        \
  "vmov.i16   q0, #0\n"               /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"       /* leakey relu alpha */ \
  "vcge.f16   q2,    q8,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q8,    q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI2_v7_LEAKY_RELU                   \
  /* do relu */                                                 \
  "11: \n"                                                      \
  "cmp   %[vflag_act],  #3       \n"    /* check leakey relu */ \
  "bne   12f                     \n"    /* no act end */        \
  "vmov.i16   q0, #0\n"                 /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"         /* leakey relu alpha */ \
  "vcge.f16   d4,    d16,     d0    \n" /* vcgeq_f32 */         \
  "vmul.f16   d6,    d16,     d2    \n" /* vmulq_f32 */         \
  "vcge.f16   d12,   d24,    d0    \n"  /* vcgeq_f32 */         \
  "vmul.f16   d14,   d24,    d2    \n"  /* vmulq_f32 */         \
  "vbif       d16,   d6,    d4    \n"                           \
  "vbif       d24,   d14,   d12    \n"                          \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_SEMI1_v7_LEAKY_RELU                   \
  /* do relu */                                                 \
  "11: \n"                                                      \
  "cmp   %[vflag_act],  #3       \n"    /* check leakey relu */ \
  "bne   12f                     \n"    /* no act end */        \
  "vmov.i16   q0, #0\n"                 /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"         /* leakey relu alpha */ \
  "vcge.f16   d4,    d16,     d0    \n" /* vcgeq_f32 */         \
  "vmul.f16   d6,    d16,     d2    \n" /* vmulq_f32 */         \
  "vbif       d16,   d6,    d4    \n"                           \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_SEMI2_v7_HARD_SWISH                       \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4,  q8, q0               \n"                          \
  "vadd.f16   q5,  q9, q0               \n"                          \
  "vadd.f16   q6,  q12, q0               \n"                         \
  "vadd.f16   q7,  q13, q0               \n"                         \
  "vmul.f16   q8,  q8, q1              \n"                           \
  "vmul.f16   q9,  q9, q1              \n"                           \
  "vmul.f16   q12, q12, q1              \n"                          \
  "vmul.f16   q13, q13, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmax.f16   q6, q6, q3                \n"                          \
  "vmax.f16   q7, q7, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmin.f16   q6, q6, q2                \n"                          \
  "vmin.f16   q7, q7, q2                \n"                          \
  "vmul.f16   q8, q8, q4              \n"                            \
  "vmul.f16   q9, q9, q5              \n"                            \
  "vmul.f16   q12, q12, q6              \n"                          \
  "vmul.f16   q13, q13, q7              \n"                          \
  "vadd.f16   q4,  q10, q0               \n"                         \
  "vadd.f16   q5,  q11, q0               \n"                         \
  "vadd.f16   q6,  q14, q0               \n"                         \
  "vadd.f16   q7,  q15, q0               \n"                         \
  "vmul.f16   q10, q10, q1              \n"                          \
  "vmul.f16   q11, q11, q1              \n"                          \
  "vmul.f16   q14, q14, q1              \n"                          \
  "vmul.f16   q15, q15, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmax.f16   q6, q6, q3                \n"                          \
  "vmax.f16   q7, q7, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmin.f16   q6, q6, q2                \n"                          \
  "vmin.f16   q7, q7, q2                \n"                          \
  "vmul.f16   q10, q10, q4              \n"                          \
  "vmul.f16   q11, q11, q5              \n"                          \
  "vmul.f16   q14, q14, q6              \n"                          \
  "vmul.f16   q15, q15, q7              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W32_SEMI1_v7_HARD_SWISH                       \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q8, q0               \n"                           \
  "vadd.f16   q5, q9, q0               \n"                           \
  "vadd.f16   q6, q10, q0               \n"                          \
  "vadd.f16   q7, q11, q0               \n"                          \
  "vmul.f16   q8, q8, q1              \n"                            \
  "vmul.f16   q9, q9, q1              \n"                            \
  "vmul.f16   q10, q10, q1              \n"                          \
  "vmul.f16   q11, q11, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmax.f16   q6, q6, q3                \n"                          \
  "vmax.f16   q7, q7, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmin.f16   q6, q6, q2                \n"                          \
  "vmin.f16   q7, q7, q2                \n"                          \
  "vmul.f16   q8, q8, q4              \n"                            \
  "vmul.f16   q9, q9, q5              \n"                            \
  "vmul.f16   q10, q10, q6              \n"                          \
  "vmul.f16   q11, q11, q7              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W16_SEMI2_v7_HARD_SWISH                       \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4,  q8, q0               \n"                          \
  "vadd.f16   q5,  q9, q0               \n"                          \
  "vadd.f16   q6,  q12, q0               \n"                         \
  "vadd.f16   q7,  q13, q0               \n"                         \
  "vmul.f16   q8,  q8, q1              \n"                           \
  "vmul.f16   q9,  q9, q1              \n"                           \
  "vmul.f16   q12, q12, q1              \n"                          \
  "vmul.f16   q13, q13, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmax.f16   q6, q6, q3                \n"                          \
  "vmax.f16   q7, q7, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmin.f16   q6, q6, q2                \n"                          \
  "vmin.f16   q7, q7, q2                \n"                          \
  "vmul.f16   q8, q8, q4              \n"                            \
  "vmul.f16   q9, q9, q5              \n"                            \
  "vmul.f16   q12, q12, q6              \n"                          \
  "vmul.f16   q13, q13, q7              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W16_SEMI1_v7_HARD_SWISH                       \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q8, q0               \n"                           \
  "vadd.f16   q5, q9, q0               \n"                           \
  "vmul.f16   q8, q8, q1              \n"                            \
  "vmul.f16   q9, q9, q1              \n"                            \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmul.f16   q8, q8, q4              \n"                            \
  "vmul.f16   q9, q9, q5              \n"                            \
  "9:\n"

#define SPARSE_F16_F16_W8_SEMI2_v7_HARD_SWISH                        \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4,  q8, q0               \n"                          \
  "vadd.f16   q6,  q12, q0               \n"                         \
  "vmul.f16   q8,  q8, q1              \n"                           \
  "vmul.f16   q12, q12, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q6, q6, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q6, q6, q2                \n"                          \
  "vmul.f16   q8, q8, q4              \n"                            \
  "vmul.f16   q12, q12, q6              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W8_SEMI1_v7_HARD_SWISH                        \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q8, q0               \n"                           \
  "vmul.f16   q8, q8, q1              \n"                            \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmul.f16   q8, q8, q4              \n"                            \
  "9:\n"

#define SPARSE_F16_F16_W4_SEMI2_v7_HARD_SWISH                        \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   d8,  d16, d0               \n"                         \
  "vadd.f16   d12,  d24, d0               \n"                        \
  "vmul.f16   d16,  d16, d2              \n"                         \
  "vmul.f16   d24,  d24, d2              \n"                         \
  "vmax.f16   d8,  d8,  d6                \n"                        \
  "vmax.f16   d12, d12, d6                \n"                        \
  "vmin.f16   d8,  d8,  d4                \n"                        \
  "vmin.f16   d12, d12, d4                \n"                        \
  "vmul.f16   d16, d16, d8              \n"                          \
  "vmul.f16   d24, d24, d12              \n"                         \
  "9:\n"

#define SPARSE_F16_F16_W4_SEMI1_v7_HARD_SWISH                        \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vmul.f16   d16,  d16, d2              \n"                         \
  "vmax.f16   d8,  d8,  d6                \n"                        \
  "vmin.f16   d8,  d8,  d4                \n"                        \
  "vmul.f16   d16, d16, d8              \n"                          \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_F16_F16_W32_SEMI2_V7_OUT                                        \
  SPARSE_F16_F16_W32_SEMI2_v7_KERNEL SPARSE_F16_F16_W32_SEMI2_v7_RELU          \
      SPARSE_F16_F16_W32_SEMI2_v7_RELU6 SPARSE_F16_F16_W32_SEMI2_v7_LEAKY_RELU \
          SPARSE_F16_F16_W32_SEMI2_v7_HARD_SWISH                               \
      "vst1.16   {d16-d19},  [%[c_ptr1]]!\n"                                   \
      "vst1.16   {d24-d27},  [%[c_ptr2]]!\n"                                   \
      "vst1.16   {d20-d23},  [%[c_ptr1]]!\n"                                   \
      "vst1.16   {d28-d31},  [%[c_ptr2]]!\n"
#define SPARSE_F16_F16_W32_SEMI1_V7_OUT                                        \
  SPARSE_F16_F16_W32_SEMI1_v7_KERNEL SPARSE_F16_F16_W32_SEMI1_v7_RELU          \
      SPARSE_F16_F16_W32_SEMI1_v7_RELU6 SPARSE_F16_F16_W32_SEMI1_v7_LEAKY_RELU \
          SPARSE_F16_F16_W32_SEMI1_v7_HARD_SWISH                               \
      "vst1.16   {d16-d19},  [%[c_ptr]]!\n"                                    \
      "vst1.16   {d20-d23},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_F16_F16_W16_SEMI2_V7_OUT                                        \
  SPARSE_F16_F16_W16_SEMI2_v7_KERNEL SPARSE_F16_F16_W16_SEMI2_v7_RELU          \
      SPARSE_F16_F16_W16_SEMI2_v7_RELU6 SPARSE_F16_F16_W16_SEMI2_v7_LEAKY_RELU \
          SPARSE_F16_F16_W16_SEMI2_v7_HARD_SWISH                               \
      "vst1.16   {d16-d19},  [%[c_ptr1]]!\n"                                   \
      "vst1.16   {d24-d27},  [%[c_ptr2]]!\n"
#define SPARSE_F16_F16_W16_SEMI1_V7_OUT                                        \
  SPARSE_F16_F16_W16_SEMI1_v7_KERNEL SPARSE_F16_F16_W16_SEMI1_v7_RELU          \
      SPARSE_F16_F16_W16_SEMI1_v7_RELU6 SPARSE_F16_F16_W16_SEMI1_v7_LEAKY_RELU \
          SPARSE_F16_F16_W16_SEMI1_v7_HARD_SWISH                               \
      "vst1.16   {d16-d19},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F16_F16_W8_SEMI2_V7_OUT                                       \
  SPARSE_F16_F16_W8_SEMI2_v7_KERNEL SPARSE_F16_F16_W8_SEMI2_v7_RELU          \
      SPARSE_F16_F16_W8_SEMI2_v7_RELU6 SPARSE_F16_F16_W8_SEMI2_v7_LEAKY_RELU \
          SPARSE_F16_F16_W8_SEMI2_v7_HARD_SWISH                              \
      "vst1.16   {d16-d17},  [%[c_ptr1]]!\n"                                 \
      "vst1.16   {d24-d25},  [%[c_ptr2]]!\n"
#define SPARSE_F16_F16_W8_SEMI1_V7_OUT                                       \
  SPARSE_F16_F16_W8_SEMI1_v7_KERNEL SPARSE_F16_F16_W8_SEMI1_v7_RELU          \
      SPARSE_F16_F16_W8_SEMI1_v7_RELU6 SPARSE_F16_F16_W8_SEMI1_v7_LEAKY_RELU \
          SPARSE_F16_F16_W8_SEMI1_v7_HARD_SWISH                              \
      "vst1.16   {d16-d17},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_F16_F16_W4_SEMI2_V7_OUT                                       \
  SPARSE_F16_F16_W4_SEMI2_v7_KERNEL SPARSE_F16_F16_W4_SEMI2_v7_RELU          \
      SPARSE_F16_F16_W4_SEMI2_v7_RELU6 SPARSE_F16_F16_W4_SEMI2_v7_LEAKY_RELU \
          SPARSE_F16_F16_W4_SEMI2_v7_HARD_SWISH                              \
      "vst1.16   {d16},  [%[c_ptr1]]!\n"                                     \
      "vst1.16   {d24},  [%[c_ptr2]]!\n"
#define SPARSE_F16_F16_W4_SEMI1_V7_OUT                                       \
  SPARSE_F16_F16_W4_SEMI1_v7_KERNEL SPARSE_F16_F16_W4_SEMI1_v7_RELU          \
      SPARSE_F16_F16_W4_SEMI1_v7_RELU6 SPARSE_F16_F16_W4_SEMI1_v7_LEAKY_RELU \
          SPARSE_F16_F16_W4_SEMI1_v7_HARD_SWISH                              \
      "vst1.16   {d16-d17},  [%[c_ptr]]!\n"

#define SET_ASSM_INPUT_PARAM1_V7_F16        \
  : [a_ptr] "+r"(cur_w),                  \
    [b_ptr] "+r"(cur_b),                  \
    [c_ptr1] "+r"(out_ptr1),              \
    [c_ptr2] "+r"(out_ptr2),              \
    [k] "+r"(nnz),                        \
    [widx_dmap] "+r"(dmap),               \
    [bias_ptr] "+r"(pbias),               \
    [hs_param] "+r"(vhs_param)             \
  : [vflag_act] "r"(flag_act),            \
    [valpha] "r"(alpha) \
  : "q0", "q1", "q2", "q3", "q4", "q5", \
    "q6", "q7", "q8", "q9", "q10", "q11", \
    "q12", "q13", "q14", "q15", "r0", "r2", \
    "cc", "memory"

#define SET_ASSM_INPUT_PARAM2_V7_F16          \
  : [a_ptr] "+r"(cur_w),                  \
    [b_ptr] "+r"(cur_b),                  \
    [c_ptr] "+r"(out_ptr),                \
    [k] "+r"(nnz),                        \
    [widx_dmap] "+r"(dmap),               \
    [hs_param] "+r"(vhs_param)             \
  : [vbias] "r"(vbias),                   \
    [vflag_act] "r"(flag_act),            \
    [valpha] "r"(alpha) \
  : "q0", "q1", "q2", "q3", "q4", "q5", \
      "q6", "q7", "q8", "q9", "q10", "q11", \
      "q12", "q13", "q14", "q15", "r0", "r2", \
      "cc", "memory"

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are f16.
 * Sparse matrix multiplication is calculated in blocks, the block size is Mx48,
 * that is,
 * the parameter matrix is MxK, and the activation matrix is Kx48; when N is
 * less than 48,
 * it is calculated in blocks of Mx32, Mx16, Mx8, and Mx4 in turn;
 * @param A sparse weight data
 * @param B dense input data
 * @param widx_dmap An array of int32_t values storing scaled [by sizeof(input
 * element)] difference
 * between input channels corresponding to successive non-zero element
 * @param nidx_nnzmap the number of non-zero kernel elements per each output
 * channel
 * @param bias
 * @param output
 * @param M
 * @param N
 * @param K
 * @param param
 * @param ctx
 */
void sparse_semi_conv_fp16_pipelined(const float16_t* A,
                                     const float16_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float16_t* bias,
                                     float16_t* output,
                                     const int M,
                                     const int K,
                                     const int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  INIT_SEMI_CONV_PARAM(float16_t, float16_t, 48)
  float16_t bias_zero[2] = {0.f, 0.f};
  while
    SPARSE_FP16_LIKELY(mc >= 32 * sizeof(float16_t)) {
      LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
        GET_SEMI_PARAM_TABLE(float16_t, float16_t)
        auto vhs_param = hs_param;
        // clang-format off
            asm volatile(SPARSE_F16_F16_W32_SEMI2_V7_OUT  
              SET_ASSM_INPUT_PARAM1_V7_F16);
        // clang-format on
      }
      LITE_PARALLEL_COMMON_END();
      if
        SPARSE_FP16_UNLIKELY(lave_num != 0) {
          GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
          auto vhs_param = hs_param;
          // clang-format off
          asm volatile(SPARSE_F16_F16_W32_SEMI1_V7_OUT  
            SET_ASSM_INPUT_PARAM2_V7_F16);
          // clang-format on
        }
      output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                            32 * sizeof(float16_t));
      B += 32;
      mc -= 32 * sizeof(float16_t);
    }

  if
    SPARSE_FP16_UNLIKELY(mc != 0) {
      if (mc >= (16 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          auto vhs_param = hs_param;
          // clang-format off
              asm volatile(SPARSE_F16_F16_W16_SEMI2_V7_OUT  
                SET_ASSM_INPUT_PARAM1_V7_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            auto vhs_param = hs_param;
            // clang-format off
            asm volatile(SPARSE_F16_F16_W16_SEMI1_V7_OUT  
              SET_ASSM_INPUT_PARAM2_V7_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              16 * sizeof(float16_t));
        B += 16;
        mc -= 16 * sizeof(float16_t);
      }
      if (mc >= (8 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          auto vhs_param = hs_param;
          // clang-format off
              asm volatile(SPARSE_F16_F16_W8_SEMI2_V7_OUT  
                SET_ASSM_INPUT_PARAM1_V7_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            auto vhs_param = hs_param;
            // clang-format off
            asm volatile(SPARSE_F16_F16_W8_SEMI1_V7_OUT  
              SET_ASSM_INPUT_PARAM2_V7_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              8 * sizeof(float16_t));
        B += 8;
        mc -= 8 * sizeof(float16_t);
      }
      if (mc >= (4 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          auto vhs_param = hs_param;
          // clang-format off
              asm volatile(SPARSE_F16_F16_W4_SEMI2_V7_OUT  
                SET_ASSM_INPUT_PARAM1_V7_F16);
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            auto vhs_param = hs_param;
            // clang-format off
            asm volatile(SPARSE_F16_F16_W4_SEMI1_V7_OUT  
              SET_ASSM_INPUT_PARAM2_V7_F16);
            // clang-format on
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              4 * sizeof(float16_t));
        B += 4;
        mc -= 4 * sizeof(float16_t);
      }
      while (mc >= (1 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, pair_num, 0, 1) {
          GET_SEMI_PARAM_TABLE(float16_t, float16_t)
          float16_t vacc01n0 = pbias[0];
          float16_t vacc01n1 = pbias[1];
          if
            SPARSE_FP16_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                vacc01n0 += cur_b[0] * cur_w[0];
                vacc01n1 += cur_b[0] * cur_w[1];
                cur_w += 2;
                cur_b = (const float16_t*)((uintptr_t)cur_b + (uintptr_t)diff);
              } while (--nnz != 0);
            }

          COMPUTE_ACT_NEON_TWO_V8_F32
          out_ptr1[0] = vacc01n0;
          out_ptr2[0] = vacc01n1;
        }
        LITE_PARALLEL_COMMON_END();
        if
          SPARSE_FP16_UNLIKELY(lave_num != 0) {
            GET_UNSTRUCT_PARAM_TABLE(float16_t, float16_t)
            float16_t vacc01n0 = vbias;
            if
              SPARSE_FP16_LIKELY(nnz != 0) {
                do {
                  const intptr_t diff = *dmap++;
                  vacc01n0 += cur_b[0] * cur_w[0];
                  cur_b =
                      (const float16_t*)((uintptr_t)cur_b + (uintptr_t)diff);
                  cur_w += 1;
                } while (--nnz != 0);
              }
            COMPUTE_ACT_NEON_ONE_V8_F32
            out_ptr[0] = vacc01n0;
          }
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              1 * sizeof(float16_t));
        B += 1;
        mc -= 1 * sizeof(float16_t);
      }
    }
}
#endif

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
