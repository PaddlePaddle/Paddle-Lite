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

#include "lite/backends/arm/math/fp16/sparse_conv_fp16.h"
#include <arm_neon.h>
#include <vector>
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

#define SPARSE_COMPUTE_LOOP                                                 \
  float16_t* cur_output =                                                   \
      reinterpret_cast<float16_t*>((uintptr_t)output + output_stride * i);  \
  const float16_t* cur_w = A;                                               \
  uint32_t nnz = nidx_nnzmap[i];                                            \
  const float16_t* cur_b = B;                                               \
  const int32_t* dmap = widx_dmap;                                          \
  if (i != 0) {                                                             \
    int cur_rem = nidx_nnzmap[i - 1] & 3;                                   \
    if (cur_rem != 0) {                                                     \
      cur_rem = 4 - cur_rem;                                                \
    }                                                                       \
    nnz = nidx_nnzmap[i] - nidx_nnzmap[i - 1] - cur_rem;                    \
    cur_w = A + nidx_nnzmap[i - 1] + cur_rem;                               \
    cur_b += ((nidx_nnzmap[i - 1] == 0)                                     \
                  ? 0                                                       \
                  : widx_dmap[nidx_nnzmap[i - 1] - 1] / sizeof(float16_t)); \
    dmap = widx_dmap + nidx_nnzmap[i - 1] + cur_rem;                        \
  }                                                                         \
  uint32_t pair_num = nnz / 4;                                              \
  uint32_t lave_num = nnz % 4;                                              \
  float16_t vbias = (bias != nullptr) ? bias[i] : 0.0;

#ifdef __aarch64__
#define SPARSE_F16_F16_W96_V8_KERNEL        \
  "dup     v20.8h,  %w[vbias]\n"            \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v22.8h,  v20.h[0]\n"             \
  "dup     v23.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v24.8h,  v20.h[0]\n"             \
  "dup     v25.8h,  v20.h[0]\n"             \
  "dup     v26.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "dup     v27.8h,  v20.h[0]\n"             \
  "dup     v28.8h,  v20.h[0]\n"             \
  "dup     v29.8h,  v20.h[0]\n"             \
  "dup     v30.8h,  v20.h[0]\n"             \
  "dup     v31.8h,  v20.h[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[0]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v23.8h,  v5.8h,  v0.h[0]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v25.8h,  v7.8h,  v0.h[0]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v26.8h,  v8.8h,  v0.h[0]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v27.8h,  v9.8h,  v0.h[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v28.8h,  v10.8h,  v0.h[0]\n"     \
  "fmla    v29.8h,  v11.8h,  v0.h[0]\n"     \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v30.8h,  v12.8h,  v0.h[0]\n"     \
  "mov   w1, v1.s[1]\n"                     \
  "fmla    v31.8h,  v13.8h,  v0.h[0]\n"     \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.8h,  v4.8h,  v0.h[1]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[1]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v24.8h,  v6.8h,  v0.h[1]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[1]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v26.8h,  v8.8h,  v0.h[1]\n"      \
  "fmla    v27.8h,  v9.8h,  v0.h[1]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v28.8h,  v10.8h,  v0.h[1]\n"     \
  "fmla    v29.8h,  v11.8h,  v0.h[1]\n"     \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v30.8h,  v12.8h,  v0.h[1]\n"     \
  "fmla    v31.8h,  v13.8h,  v0.h[1]\n"     \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.8h,  v4.8h,  v0.h[2]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[2]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v24.8h,  v6.8h,  v0.h[2]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[2]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v26.8h,  v8.8h,  v0.h[2]\n"      \
  "fmla    v27.8h,  v9.8h,  v0.h[2]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v28.8h,  v10.8h,  v0.h[2]\n"     \
  "fmla    v29.8h,  v11.8h,  v0.h[2]\n"     \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v30.8h,  v12.8h,  v0.h[2]\n"     \
  "mov   w1, v1.s[3]\n"                     \
  "fmla    v31.8h,  v13.8h,  v0.h[2]\n"     \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[3]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v22.8h,  v4.8h,  v0.h[3]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v24.8h,  v6.8h,  v0.h[3]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[3]\n"      \
  "fmla    v26.8h,  v8.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v27.8h,  v9.8h,  v0.h[3]\n"      \
  "fmla    v28.8h,  v10.8h,  v0.h[3]\n"     \
  "fmla    v29.8h,  v11.8h,  v0.h[3]\n"     \
  "fmla    v30.8h,  v12.8h,  v0.h[3]\n"     \
  "fmla    v31.8h,  v13.8h,  v0.h[3]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[0]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v23.8h,  v5.8h,  v0.h[0]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v25.8h,  v7.8h,  v0.h[0]\n"      \
  "fmla    v26.8h,  v8.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v27.8h,  v9.8h,  v0.h[0]\n"      \
  "fmla    v28.8h,  v10.8h,  v0.h[0]\n"     \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v29.8h,  v11.8h,  v0.h[0]\n"     \
  "fmla    v30.8h,  v12.8h,  v0.h[0]\n"     \
  "fmla    v31.8h,  v13.8h,  v0.h[0]\n"     \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.8h,  v4.8h,  v0.h[1]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[1]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[1]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v25.8h,  v7.8h,  v0.h[1]\n"      \
  "fmla    v26.8h,  v8.8h,  v0.h[1]\n"      \
  "fmla    v27.8h,  v9.8h,  v0.h[1]\n"      \
  "fmla    v28.8h,  v10.8h,  v0.h[1]\n"     \
  "fmla    v29.8h,  v11.8h,  v0.h[1]\n"     \
  "fmla    v30.8h,  v12.8h,  v0.h[1]\n"     \
  "fmla    v31.8h,  v13.8h,  v0.h[1]\n"     \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[2]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v23.8h,  v5.8h,  v0.h[2]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[2]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v25.8h,  v7.8h,  v0.h[2]\n"      \
  "fmla    v26.8h,  v8.8h,  v0.h[2]\n"      \
  "fmla    v27.8h,  v9.8h,  v0.h[2]\n"      \
  "fmla    v28.8h,  v10.8h,  v0.h[2]\n"     \
  "fmla    v29.8h,  v11.8h,  v0.h[2]\n"     \
  "fmla    v30.8h,  v12.8h,  v0.h[2]\n"     \
  "fmla    v31.8h,  v13.8h,  v0.h[2]\n"     \
  "1:\n"

#define SPARSE_F16_F16_W48_V8_KERNEL        \
  "dup     v20.8h,  %w[vbias]\n"            \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v22.8h,  v20.h[0]\n"             \
  "dup     v23.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v24.8h,  v20.h[0]\n"             \
  "dup     v25.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v22.8h,  v4.8h,  v0.h[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.8h,  v5.8h,  v0.h[0]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v24.8h,  v6.8h,  v0.h[0]\n"      \
  "mov   w1, v1.s[1]\n"                     \
  "fmla    v25.8h,  v7.8h,  v0.h[0]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.8h,  v5.8h,  v0.h[1]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[1]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v25.8h,  v7.8h,  v0.h[1]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.8h,  v4.8h,  v0.h[2]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[2]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v24.8h,  v6.8h,  v0.h[2]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[2]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "mov   w1, v1.s[3]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[3]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[3]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[3]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #64]\n"  \
  "fmla    v25.8h,  v7.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v23.8h,  v5.8h,  v0.h[0]\n"      \
  "fmla    v24.8h,  v6.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.8h,  v4.8h,  v0.h[1]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[1]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v24.8h,  v6.8h,  v0.h[1]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[1]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.8h,  v4.8h,  v0.h[2]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[2]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v24.8h,  v6.8h,  v0.h[2]\n"      \
  "fmla    v25.8h,  v7.8h,  v0.h[2]\n"      \
  "1:\n"

#define SPARSE_F16_F16_W32_V8_KERNEL        \
  "dup     v20.8h,  %w[vbias]\n"            \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v22.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v23.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v22.8h,  v4.8h,  v0.h[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.8h,  v5.8h,  v0.h[0]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.8h,  v4.8h,  v0.h[1]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[1]\n"      \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.8h,  v4.8h,  v0.h[2]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[2]\n"      \
  "mov   w1, v1.s[3]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[3]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v23.8h,  v5.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]], #16\n"         \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "fmla    v22.8h,  v4.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v23.8h,  v5.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #64]\n"  \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.8h,  v4.8h,  v0.h[1]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[1]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.8h,  v4.8h,  v0.h[2]\n"      \
  "fmla    v23.8h,  v5.8h,  v0.h[2]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "1:\n"

#define SPARSE_F16_F16_W16_V8_KERNEL        \
  "dup     v20.8h,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "mov   w1, v1.s[3]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[3]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[3]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.8h,  v3.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "fmla    v21.8h,  v3.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "1:\n"

#define SPARSE_F16_F16_W8_V8_KERNEL         \
  "dup     v20.8h,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v21.8h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "mov   w1, v1.s[1]\n"                     \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "mov   w1, v1.s[2]\n"                     \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "mov   w1, v1.s[3]\n"                     \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[3]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #64]\n"  \
  "beq     1f\n"                            \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.8h,  v2.8h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "beq     1f\n"                            \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.8h,  v2.8h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "1:\n"

#define SPARSE_F16_F16_W4_V8_KERNEL         \
  "dup     v20.4h,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "dup     v21.4h,  v20.h[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.4h,  v2.4h,  v0.h[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "mov   w1, v1.s[1]\n"                     \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.4h,  v2.4h,  v0.h[1]\n"      \
  "mov   w1, v1.s[2]\n"                     \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.4h,  v2.4h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "mov   w1, v1.s[3]\n"                     \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.4h,  v2.4h,  v0.h[3]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   d0, [%[a_ptr]], #8\n"              \
  "ldr   q1, [%[widx_dmap]],  #16\n"        \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.4h,  v2.4h,  v0.h[0]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "prfm  pldl1keep, [%[a_ptr], #64]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #64]\n"  \
  "beq     1f\n"                            \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v20.4h,  v2.4h,  v0.h[1]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "beq     1f\n"                            \
  "ldr   d2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "fmla    v20.4h,  v2.4h,  v0.h[2]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "1:\n"

#define SPARSE_F16_F16_W96_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v0.8h, #0\n"               \
  "fmax   v20.8h, v20.8h, v0.8h\n"   \
  "fmax   v21.8h, v21.8h, v0.8h\n"   \
  "fmax   v22.8h, v22.8h, v0.8h\n"   \
  "fmax   v23.8h, v23.8h, v0.8h\n"   \
  "fmax   v24.8h, v24.8h, v0.8h\n"   \
  "fmax   v25.8h, v25.8h, v0.8h\n"   \
  "fmax   v26.8h, v26.8h, v0.8h\n"   \
  "fmax   v27.8h, v27.8h, v0.8h\n"   \
  "fmax   v28.8h, v28.8h, v0.8h\n"   \
  "fmax   v29.8h, v29.8h, v0.8h\n"   \
  "fmax   v30.8h, v30.8h, v0.8h\n"   \
  "fmax   v31.8h, v31.8h, v0.8h\n"   \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v30.8h, #0\n"              \
  "fmax   v20.8h, v20.8h, v30.8h\n"  \
  "fmax   v21.8h, v21.8h, v30.8h\n"  \
  "fmax   v22.8h, v22.8h, v30.8h\n"  \
  "fmax   v23.8h, v23.8h, v30.8h\n"  \
  "fmax   v24.8h, v24.8h, v30.8h\n"  \
  "fmax   v25.8h, v25.8h, v30.8h\n"  \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.8h, #0\n"               \
  "fmax   v20.8h, v20.8h, v9.8h\n"   \
  "fmax   v21.8h, v21.8h, v9.8h\n"   \
  "fmax   v22.8h, v22.8h, v9.8h\n"   \
  "fmax   v23.8h, v23.8h, v9.8h\n"   \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.8h, #0\n"               \
  "fmax   v20.8h, v20.8h, v9.8h\n"   \
  "fmax   v21.8h, v21.8h, v9.8h\n"   \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_V8_RELU    \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.8h, #0\n"               \
  "fmax   v20.8h, v20.8h, v9.8h\n"   \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_V8_RELU    \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.4h, #0\n"               \
  "fmax   v20.4h, v20.4h, v9.4h\n"   \
  "b      9f                    \n"

#define SPARSE_F16_F16_W96_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.8h, #0\n"                \
  "dup    v1.8h,  %w[valpha]\n"       \
  "fmax   v20.8h, v20.8h, v0.8h\n"    \
  "fmax   v21.8h, v21.8h, v0.8h\n"    \
  "fmax   v22.8h, v22.8h, v0.8h\n"    \
  "fmax   v23.8h, v23.8h, v0.8h\n"    \
  "fmax   v24.8h, v24.8h, v0.8h\n"    \
  "fmax   v25.8h, v25.8h, v0.8h\n"    \
  "fmax   v26.8h, v26.8h, v0.8h\n"    \
  "fmax   v27.8h, v27.8h, v0.8h\n"    \
  "fmax   v28.8h, v28.8h, v0.8h\n"    \
  "fmax   v29.8h, v29.8h, v0.8h\n"    \
  "fmax   v30.8h, v30.8h, v0.8h\n"    \
  "fmax   v31.8h, v31.8h, v0.8h\n"    \
  "fmin   v20.8h, v20.8h, v1.8h\n"    \
  "fmin   v21.8h, v21.8h, v1.8h\n"    \
  "fmin   v22.8h, v22.8h, v1.8h\n"    \
  "fmin   v23.8h, v23.8h, v1.8h\n"    \
  "fmin   v24.8h, v24.8h, v1.8h\n"    \
  "fmin   v25.8h, v25.8h, v1.8h\n"    \
  "fmin   v26.8h, v26.8h, v1.8h\n"    \
  "fmin   v27.8h, v27.8h, v1.8h\n"    \
  "fmin   v28.8h, v28.8h, v1.8h\n"    \
  "fmin   v29.8h, v29.8h, v1.8h\n"    \
  "fmin   v30.8h, v30.8h, v1.8h\n"    \
  "fmin   v31.8h, v31.8h, v1.8h\n"    \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.8h, #0\n"                \
  "dup    v1.8h,  %w[valpha]\n"       \
  "fmax   v20.8h, v20.8h, v0.8h\n"    \
  "fmax   v21.8h, v21.8h, v0.8h\n"    \
  "fmax   v22.8h, v22.8h, v0.8h\n"    \
  "fmax   v23.8h, v23.8h, v0.8h\n"    \
  "fmax   v24.8h, v24.8h, v0.8h\n"    \
  "fmax   v25.8h, v25.8h, v0.8h\n"    \
  "fmin   v20.8h, v20.8h, v1.8h\n"    \
  "fmin   v21.8h, v21.8h, v1.8h\n"    \
  "fmin   v22.8h, v22.8h, v1.8h\n"    \
  "fmin   v23.8h, v23.8h, v1.8h\n"    \
  "fmin   v24.8h, v24.8h, v1.8h\n"    \
  "fmin   v25.8h, v25.8h, v1.8h\n"    \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.8h, #0\n"                \
  "dup    v1.8h,  %w[valpha]\n"       \
  "fmax   v20.8h, v20.8h, v0.8h\n"    \
  "fmax   v21.8h, v21.8h, v0.8h\n"    \
  "fmax   v22.8h, v22.8h, v0.8h\n"    \
  "fmax   v23.8h, v23.8h, v0.8h\n"    \
  "fmin   v20.8h, v20.8h, v1.8h\n"    \
  "fmin   v21.8h, v21.8h, v1.8h\n"    \
  "fmin   v22.8h, v22.8h, v1.8h\n"    \
  "fmin   v23.8h, v23.8h, v1.8h\n"    \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.8h, #0\n"                \
  "dup    v1.8h,  %w[valpha]\n"       \
  "fmax   v20.8h, v20.8h, v0.8h\n"    \
  "fmax   v21.8h, v21.8h, v0.8h\n"    \
  "fmin   v20.8h, v20.8h, v1.8h\n"    \
  "fmin   v21.8h, v21.8h, v1.8h\n"    \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_V8_RELU6    \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.8h, #0\n"                \
  "dup    v1.8h,  %w[valpha]\n"       \
  "fmax   v20.8h, v20.8h, v0.8h\n"    \
  "fmin   v20.8h, v20.8h, v1.8h\n"    \
  "b      9f                    \n"
#define SPARSE_F16_F16_W4_V8_RELU6    \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.4h, #0\n"                \
  "dup    v1.4h,  %w[valpha]\n"       \
  "fmax   v20.4h, v20.4h, v0.4h\n"    \
  "fmin   v20.4h, v20.4h, v1.4h\n"    \
  "b      9f                    \n"

#define SPARSE_F16_F16_W96_V8_LEAKY_RELU                            \
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

#define SPARSE_F16_F16_W48_V8_LEAKY_RELU                           \
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

#define SPARSE_F16_F16_W32_V8_LEAKY_RELU                           \
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

#define SPARSE_F16_F16_W16_V8_LEAKY_RELU                           \
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

#define SPARSE_F16_F16_W8_V8_LEAKY_RELU                            \
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

#define SPARSE_F16_F16_W4_V8_LEAKY_RELU                            \
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

#define SPARSE_F16_F16_W96_V8_HARD_SWISH                         \
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

#define SPARSE_F16_F16_W48_V8_HARD_SWISH                         \
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

#define SPARSE_F16_F16_W32_V8_HARD_SWISH                         \
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

#define SPARSE_F16_F16_W16_V8_HARD_SWISH                         \
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

#define SPARSE_F16_F16_W8_V8_HARD_SWISH                          \
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

#define SPARSE_F16_F16_W4_V8_HARD_SWISH                          \
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
 * The data block size for sparse matrix calculation is Mx96, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx96.
 */
#define SPARSE_F16_F16_W96_V8_OUT       \
  SPARSE_F16_F16_W96_V8_KERNEL          \
  SPARSE_F16_F16_W96_V8_RELU            \
  SPARSE_F16_F16_W96_V8_RELU6           \
  SPARSE_F16_F16_W96_V8_LEAKY_RELU      \
  SPARSE_F16_F16_W96_V8_HARD_SWISH      \
  /* store result */                    \
  "stp   q20, q21,  [%[c_ptr]]\n"       \
  "stp   q22, q23,  [%[c_ptr], #32]\n"  \
  "stp   q24, q25,  [%[c_ptr], #64]\n"  \
  "stp   q26, q27,  [%[c_ptr], #96]\n"  \
  "stp   q28, q29,  [%[c_ptr], #128]\n" \
  "stp   q30, q31,  [%[c_ptr], #160]\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_F16_F16_W48_V8_OUT      \
  SPARSE_F16_F16_W48_V8_KERNEL         \
  SPARSE_F16_F16_W48_V8_RELU           \
  SPARSE_F16_F16_W48_V8_RELU6          \
  SPARSE_F16_F16_W48_V8_LEAKY_RELU     \
  SPARSE_F16_F16_W48_V8_HARD_SWISH     \
  /* store result */                   \
  "stp   q20, q21,  [%[c_ptr]]\n"      \
  "stp   q22, q23,  [%[c_ptr], #32]\n" \
  "stp   q24, q25,  [%[c_ptr], #64]\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx32.
 */
#define SPARSE_F16_F16_W32_V8_OUT  \
  SPARSE_F16_F16_W32_V8_KERNEL     \
  SPARSE_F16_F16_W32_V8_RELU       \
  SPARSE_F16_F16_W32_V8_RELU6      \
  SPARSE_F16_F16_W32_V8_LEAKY_RELU \
  SPARSE_F16_F16_W32_V8_HARD_SWISH \
  /* store result */               \
  "stp   q20, q21,  [%[c_ptr]]\n"  \
  "stp   q22, q23,  [%[c_ptr], #32]\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_F16_F16_W16_V8_OUT  \
  SPARSE_F16_F16_W16_V8_KERNEL     \
  SPARSE_F16_F16_W16_V8_RELU       \
  SPARSE_F16_F16_W16_V8_RELU6      \
  SPARSE_F16_F16_W16_V8_LEAKY_RELU \
  SPARSE_F16_F16_W16_V8_HARD_SWISH \
  /* store result */               \
  "stp   q20, q21,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F16_F16_W8_V8_OUT  \
  SPARSE_F16_F16_W8_V8_KERNEL     \
  SPARSE_F16_F16_W8_V8_RELU       \
  SPARSE_F16_F16_W8_V8_RELU6      \
  SPARSE_F16_F16_W8_V8_LEAKY_RELU \
  SPARSE_F16_F16_W8_V8_HARD_SWISH \
  /* store result */              \
  "str   q20,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_F16_F16_W4_V8_OUT  \
  SPARSE_F16_F16_W4_V8_KERNEL     \
  SPARSE_F16_F16_W4_V8_RELU       \
  SPARSE_F16_F16_W4_V8_RELU6      \
  SPARSE_F16_F16_W4_V8_LEAKY_RELU \
  SPARSE_F16_F16_W4_V8_HARD_SWISH \
  /* store result */              \
  "str   d20,  [%[c_ptr]]\n"

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are f16.
 * Sparse matrix multiplication is calculated in blocks, the block size is Mx96,
 * that is,
 * the parameter matrix is MxK, and the activation matrix is Kx96; when N is
 * less than 96,
 * it is calculated in blocks of Mx48, Mx32, Mx16, Mx8, and Mx4 in turn;
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
void sparse_conv_fp16_pipelined(const float16_t* A,
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
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  volatile float16_t alpha = 0.f;
  float16_t hs_param[24] = {0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3, hard_swish: 4
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = static_cast<float16_t>(act_param.Relu_clipped_coef);
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = static_cast<float16_t>(act_param.Leaky_relu_alpha);
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 0x04;
      for (int i = 0; i < 8; i++) {
        hs_param[i] = static_cast<float16_t>(act_param.hard_swish_offset);
        hs_param[i + 8] =
            static_cast<float16_t>(1.0 / act_param.hard_swish_scale);
        hs_param[i + 16] =
            static_cast<float16_t>(act_param.hard_swish_threshold);
      }
    }
  }
  int flag_bias = (bias != nullptr) ? 1 : 0;
  size_t mc = N * sizeof(float16_t);
  size_t nc = M;
  size_t output_stride = N * sizeof(float16_t);
  size_t output_decrement = output_stride * nc - 96 * sizeof(float16_t);
  while
    SPARSE_FP16_LIKELY(mc >= 96 * sizeof(float16_t)) {
      LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
        SPARSE_COMPUTE_LOOP
        // clang-format off
            asm volatile(SPARSE_F16_F16_W96_V8_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha),
                [hs_param] "r"(hs_param)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w1", "x1", "cc", "memory");
        // clang-format on
      }
      LITE_PARALLEL_COMMON_END();
      output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                            96 * sizeof(float16_t));
      B += 96;
      mc -= 96 * sizeof(float16_t);
    }
  if
    SPARSE_FP16_UNLIKELY(mc != 0) {
      if (mc >= (48 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          // clang-format off
            asm volatile(SPARSE_F16_F16_W48_V8_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha),
                [hs_param] "r"(hs_param)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", 
                "w1", "x1", "cc", "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              48 * sizeof(float16_t));
        B += 48;
        mc -= 48 * sizeof(float16_t);
      }
      if (mc >= (32 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          // clang-format off
            asm volatile(SPARSE_F16_F16_W32_V8_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha),
                [hs_param] "r"(hs_param)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v20", "v21", "v22", "v23", "w1", "x1", "cc", "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              32 * sizeof(float16_t));
        B += 32;
        mc -= 32 * sizeof(float16_t);
      }
      if (mc >= (16 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          // clang-format off
            asm volatile(SPARSE_F16_F16_W16_V8_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha),
                [hs_param] "r"(hs_param)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v11", "v12", "v20", 
              "v21", "w1", "x1", "cc", "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              16 * sizeof(float16_t));
        B += 16;
        mc -= 16 * sizeof(float16_t);
      }
      if (mc >= (8 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          // clang-format off
            asm volatile(SPARSE_F16_F16_W8_V8_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha),
                [hs_param] "r"(hs_param)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v11", "v12", "v20", 
              "v21", "w1", "x1", "cc", "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              8 * sizeof(float16_t));
        B += 8;
        mc -= 8 * sizeof(float16_t);
      }
      if (mc >= (4 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          // clang-format off
            asm volatile(SPARSE_F16_F16_W4_V8_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha),
                [hs_param] "r"(hs_param)
              : "v0", "v1", "v2", "v3", "v4", "v6", "v7", "v9", "v11", "v20", 
              "w1", "x1", "cc", "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              4 * sizeof(float16_t));
        B += 4;
        mc -= 4 * sizeof(float16_t);
      }
      while (mc >= (1 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          float16_t vout0 = vbias;
          if
            SPARSE_FP16_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                vout0 += cur_b[0] * cur_w[0];
                cur_b = (const float16_t*)((uintptr_t)cur_b + (uintptr_t)diff);
                cur_w += 1;
              } while (--nnz != 0);
            }
          if (flag_act == 1) {
            vout0 = vout0 > 0.f ? vout0 : 0.f;
          } else if (flag_act == 0) {
          } else if (flag_act == 2) {
            vout0 = vout0 > 0.f ? vout0 : 0.f;
            vout0 = vout0 > alpha ? alpha : vout0;
          } else if (flag_act == 3) {
            vout0 = vout0 > 0.f ? vout0 : (vout0 * alpha);
          } else if (flag_act == 4) {
            auto tmp0 =
                vout0 + static_cast<float16_t>(act_param.hard_swish_offset);
            auto tmp1 =
                vout0 / static_cast<float16_t>(act_param.hard_swish_scale);
            tmp0 = tmp0 > 0.f ? (tmp0 < static_cast<float16_t>(
                                            act_param.hard_swish_threshold)
                                     ? tmp0
                                     : static_cast<float16_t>(
                                           act_param.hard_swish_threshold))
                              : 0.f;
            vout0 = tmp0 * tmp1;
          } else {
            LOG(FATAL) << "This act: " << flag_act << " doesn't support";
          }
          cur_output[0] = vout0;
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              1 * sizeof(float16_t));
        B += 1;
        mc -= 1 * sizeof(float16_t);
      }
    }
}

#else  // armv7

#define SPARSE_F16_F16_W48_v7_KERNEL \
  "vdup.16    q4,    %[vbias]\n"     \
  "pld  [%[a_ptr], #64]    \n"       \
  "pld  [%[widx_dmap], #64]    \n"   \
  "vdup.16    q10,   d8[0]\n"        \
  "vdup.16    q11,   d8[0]\n"        \
  "vdup.16    q12,   d8[0]\n"        \
  "pld  [%[b_ptr], #192]    \n"      \
  "vdup.16    q13,   d8[0]\n"        \
  "vdup.16    q14,   d8[0]\n"        \
  "vdup.16    q15,   d8[0]\n"        \
  "cmp    %[n],    #0\n"             \
  "beq 3f\n"                         \
  "0:\n"                             \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "add r2, r2, #32\n"                \
  "vmla.f16    q11,   q2,  d0[0]\n"  \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q12,   q3,  d0[0]\n"  \
  "vmla.f16    q13,   q4,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q14,   q5,  d0[0]\n"  \
  "vmla.f16    q15,   q6,  d0[0]\n"  \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "subs    %[n],   #1\n"             \
  "add r2, r2, #32\n"                \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "vmla.f16    q11,   q2,  d0[1]\n"  \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q12,   q3,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q13,   q4,  d0[1]\n"  \
  "vmla.f16    q14,   q5,  d0[1]\n"  \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "vmla.f16    q15,   q6,  d0[1]\n"  \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "vmla.f16    q11,   q2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "mov   r2,   %[b_ptr]\n"           \
  "vmla.f16    q12,   q3,  d0[2]\n"  \
  "vmla.f16    q13,   q4,  d0[2]\n"  \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "vmla.f16    q14,   q5,  d0[2]\n"  \
  "vmla.f16    q15,   q6,  d0[2]\n"  \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "vmla.f16    q10,   q1,  d0[3]\n"  \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "vmla.f16    q11,   q2,  d0[3]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[3]\n"  \
  "vmla.f16    q13,   q4,  d0[3]\n"  \
  "vmla.f16    q14,   q5,  d0[3]\n"  \
  "vmla.f16    q15,   q6,  d0[3]\n"  \
  "bne     0b\n"                     \
  "3: \n"                            \
  "cmp    %[m],    #0\n"             \
  "beq 1f\n"                         \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "subs  %[m],   #1\n"               \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "vmla.f16    q11,   q2,  d0[0]\n"  \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q12,   q3,  d0[0]\n"  \
  "vmla.f16    q13,   q4,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q14,   q5,  d0[0]\n"  \
  "vmla.f16    q15,   q6,  d0[0]\n"  \
  "beq     1f\n"                     \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "vmla.f16    q11,   q2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "subs  %[m],   #1\n"               \
  "vmla.f16    q12,   q3,  d0[1]\n"  \
  "vmla.f16    q13,   q4,  d0[1]\n"  \
  "vmla.f16    q14,   q5,  d0[1]\n"  \
  "vmla.f16    q15,   q6,  d0[1]\n"  \
  "beq     1f\n"                     \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "add r2, r2, #32\n"                \
  "vld1.16  {d10-d13}, [r2]\n"       \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "vmla.f16    q11,   q2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[2]\n"  \
  "vmla.f16    q13,   q4,  d0[2]\n"  \
  "vmla.f16    q14,   q5,  d0[2]\n"  \
  "vmla.f16    q15,   q6,  d0[2]\n"  \
  "1:\n"

#define SPARSE_F16_F16_W32_v7_KERNEL \
  "vdup.16    q4,    %[vbias]\n"     \
  "pld  [%[a_ptr], #64]    \n"       \
  "pld  [%[widx_dmap], #64]    \n"   \
  "vdup.16    q10,   d8[0]\n"        \
  "vdup.16    q11,   d8[0]\n"        \
  "pld  [%[b_ptr], #192]    \n"      \
  "vdup.16    q12,   d8[0]\n"        \
  "vdup.16    q13,   d8[0]\n"        \
  "cmp    %[n],    #0\n"             \
  "beq 3f\n"                         \
  "0:\n"                             \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "pld  [%[widx_dmap], #128]    \n"  \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "vmla.f16    q11,   q2,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[0]\n"  \
  "vmla.f16    q13,   q4,  d0[0]\n"  \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "subs    %[n],   #1\n"             \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "vmla.f16    q11,   q2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[1]\n"  \
  "vmla.f16    q13,   q4,  d0[1]\n"  \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "vmla.f16    q11,   q2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[2]\n"  \
  "vmla.f16    q13,   q4,  d0[2]\n"  \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "vmla.f16    q10,   q1,  d0[3]\n"  \
  "vmla.f16    q11,   q2,  d0[3]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[3]\n"  \
  "vmla.f16    q13,   q4,  d0[3]\n"  \
  "bne     0b\n"                     \
  "3: \n"                            \
  "cmp    %[m],    #0\n"             \
  "beq 1f\n"                         \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "vmla.f16    q11,   q2,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[0]\n"  \
  "vmla.f16    q13,   q4,  d0[0]\n"  \
  "beq     1f\n"                     \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "subs  %[m],   #1\n"               \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "vmla.f16    q11,   q2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[1]\n"  \
  "vmla.f16    q13,   q4,  d0[1]\n"  \
  "beq     1f\n"                     \
  "mov   r2,   %[b_ptr]\n"           \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add r2, r2, #32\n"                \
  "vld1.16  {d6-d9}, [r2]\n"         \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "vmla.f16    q11,   q2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vmla.f16    q12,   q3,  d0[2]\n"  \
  "vmla.f16    q13,   q4,  d0[2]\n"  \
  "1:\n"

#define SPARSE_F16_F16_W16_v7_KERNEL \
  "vdup.16    q4,    %[vbias]\n"     \
  "pld  [%[a_ptr], #64]    \n"       \
  "pld  [%[widx_dmap], #64]    \n"   \
  "vdup.16    q10,   d8[0]\n"        \
  "vdup.16    q11,   d8[0]\n"        \
  "pld  [%[b_ptr], #192]    \n"      \
  "cmp    %[n],    #0\n"             \
  "beq 3f\n"                         \
  "0:\n"                             \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "pld  [%[widx_dmap], #128]    \n"  \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "vmla.f16    q11,   q2,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "subs    %[n],   #1\n"             \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "vmla.f16    q11,   q2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "vmla.f16    q11,   q2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[3]\n"  \
  "vmla.f16    q11,   q2,  d0[3]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "bne     0b\n"                     \
  "3: \n"                            \
  "cmp    %[m],    #0\n"             \
  "beq 1f\n"                         \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "pld  [%[widx_dmap], #128]    \n"  \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "vmla.f16    q11,   q2,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "beq     1f\n"                     \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "vmla.f16    q11,   q2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "beq     1f\n"                     \
  "vld1.16  {d2-d5}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "vmla.f16    q11,   q2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "1:\n"

#define SPARSE_F16_F16_W8_v7_KERNEL  \
  "vdup.16    q4,    %[vbias]\n"     \
  "pld  [%[a_ptr], #64]    \n"       \
  "pld  [%[widx_dmap], #64]    \n"   \
  "vdup.16    q10,   d8[0]\n"        \
  "pld  [%[b_ptr], #192]    \n"      \
  "cmp    %[n],    #0\n"             \
  "beq 3f\n"                         \
  "0:\n"                             \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "pld  [%[widx_dmap], #128]    \n"  \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "subs   %[n],   #1\n"              \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[3]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "bne     0b\n"                     \
  "3: \n"                            \
  "cmp    %[m],    #0\n"             \
  "beq 1f\n"                         \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "pld  [%[widx_dmap], #128]    \n"  \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "beq     1f\n"                     \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "beq     1f\n"                     \
  "vld1.16  {d2-d3}, [%[b_ptr]]\n"   \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    q10,   q1,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "1:\n"

#define SPARSE_F16_F16_W4_v7_KERNEL  \
  "vdup.16    q4,    %[vbias]\n"     \
  "pld  [%[a_ptr], #64]    \n"       \
  "pld  [%[widx_dmap], #64]    \n"   \
  "vdup.16    q10,   d8[0]\n"        \
  "pld  [%[b_ptr], #192]    \n"      \
  "cmp    %[n],    #0\n"             \
  "beq 3f\n"                         \
  "0:\n"                             \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "pld  [%[widx_dmap], #128]    \n"  \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "subs    %[n],   #1\n"             \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[3]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "bne     0b\n"                     \
  "3: \n"                            \
  "cmp    %[m],    #0\n"             \
  "beq 1f\n"                         \
  "vld1.16  d0, [%[a_ptr]]\n"        \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "add %[a_ptr], %[a_ptr], #8\n"     \
  "pld  [%[widx_dmap], #128]    \n"  \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[0]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "beq     1f\n"                     \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "subs  %[m],   #1\n"               \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[1]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "beq     1f\n"                     \
  "vld1.16  {d2}, [%[b_ptr]]\n"      \
  "ldr   r0, [%[widx_dmap]],   #4\n" \
  "vmla.f16    d20,   d2,  d0[2]\n"  \
  "add   %[b_ptr],  %[b_ptr], r0\n"  \
  "1:\n"

#define SPARSE_F16_F16_W48_v7_RELU               \
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
  "vmax.f16   q14,  q14,  q0\n" /* relu */       \
  "vmax.f16   q15,  q15,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W32_v7_RELU               \
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

#define SPARSE_F16_F16_W16_v7_RELU               \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q10,  q10,  q0\n" /* relu */       \
  "vmax.f16   q11,  q11,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W8_v7_RELU                \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   q0, #0\n"         /* for relu */   \
  "vmax.f16   q10,  q10,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W4_v7_RELU                \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i16   d0, #0\n"         /* for relu */   \
  "vmax.f16   d20,  d20,  d0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F16_F16_W48_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f16   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f16   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f16   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f16   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f16   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f16   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f16   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f16   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f16   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f16   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W32_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f16   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f16   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f16   q13,  q13,  q0\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f16   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f16   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f16   q13,  q13,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W16_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f16   q11,  q11,  q0\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f16   q11,  q11,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W8_v7_RELU6                     \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   q0,   #0\n"            /* for relu6 */   \
  "vdup.16    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   q10,  q10,  q0\n"      /* relu6 */       \
  "vmin.f16   q10,  q10,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W4_v7_RELU6                     \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i16   d0,   #0\n"            /* for relu6 */   \
  "vdup.16    d1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f16   d10,  d10,  d0\n"      /* relu6 */       \
  "vmin.f16   d10,  d10,  d1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F16_F16_W48_v7_LEAKY_RELU                       \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q11,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q6,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q12,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q8,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q9,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "vbif       q11,    q5,    q4    \n"                         \
  "vbif       q12,    q7,    q6    \n"                         \
  "vbif       q13,    q9,    q8    \n"                         \
  "vcge.f16   q2,    q14,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q14,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q15,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q15,    q1    \n" /* vmulq_f32 */         \
  "vbif       q14,    q3,    q2    \n"                         \
  "vbif       q15,    q5,    q4    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W32_v7_LEAKY_RELU                       \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q11,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q6,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q7,    q12,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q8,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q9,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "vbif       q11,    q5,    q4    \n"                         \
  "vbif       q12,    q7,    q6    \n"                         \
  "vbif       q13,    q9,    q8    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W16_v7_LEAKY_RELU                       \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vcge.f16   q4,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q5,    q11,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "vbif       q11,    q5,    q4    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W8_v7_LEAKY_RELU                        \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   q0, #0\n"                /* for relu */          \
  "vdup.16    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f16   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W4_v7_LEAKY_RELU                        \
  /* do relu */                                                \
  "11: \n"                                                     \
  "cmp   %[vflag_act],  #3       \n"   /* check leakey relu */ \
  "bne   12f                     \n"   /* no act end */        \
  "vmov.i16   d0, #0\n"                /* for relu */          \
  "vdup.16    d1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f16   d2,    d20,    d0    \n" /* vcgeq_f32 */         \
  "vmul.f16   d3,    d20,    d1    \n" /* vmulq_f32 */         \
  "vbif       d20,    d3,    d2    \n"                         \
  "b      9f                    \n"

#define SPARSE_F16_F16_W48_v7_HARD_SWISH                             \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q10, q0               \n"                          \
  "vadd.f16   q5, q11, q0               \n"                          \
  "vadd.f16   q6, q12, q0               \n"                          \
  "vadd.f16   q7, q13, q0               \n"                          \
  "vmul.f16   q10, q10, q1              \n"                          \
  "vmul.f16   q11, q11, q1              \n"                          \
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
  "vmul.f16   q10, q10, q4              \n"                          \
  "vmul.f16   q11, q11, q5              \n"                          \
  "vmul.f16   q12, q12, q6              \n"                          \
  "vmul.f16   q13, q13, q7              \n"                          \
  "vadd.f16   q4, q14, q0               \n"                          \
  "vadd.f16   q5, q15, q0               \n"                          \
  "vmul.f16   q14, q14, q1              \n"                          \
  "vmul.f16   q15, q15, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmul.f16   q14, q14, q4              \n"                          \
  "vmul.f16   q15, q15, q5              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W32_v7_HARD_SWISH                             \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q10, q0               \n"                          \
  "vadd.f16   q5, q11, q0               \n"                          \
  "vadd.f16   q6, q12, q0               \n"                          \
  "vadd.f16   q7, q13, q0               \n"                          \
  "vmul.f16   q10, q10, q1              \n"                          \
  "vmul.f16   q11, q11, q1              \n"                          \
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
  "vmul.f16   q10, q10, q4              \n"                          \
  "vmul.f16   q11, q11, q5              \n"                          \
  "vmul.f16   q12, q12, q6              \n"                          \
  "vmul.f16   q13, q13, q7              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W16_v7_HARD_SWISH                             \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q10, q0               \n"                          \
  "vadd.f16   q5, q11, q0               \n"                          \
  "vmul.f16   q10, q10, q1              \n"                          \
  "vmul.f16   q11, q11, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmax.f16   q5, q5, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmin.f16   q5, q5, q2                \n"                          \
  "vmul.f16   q10, q10, q4              \n"                          \
  "vmul.f16   q11, q11, q5              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W8_v7_HARD_SWISH                              \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   q3,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   q4, q10, q0               \n"                          \
  "vmul.f16   q10, q10, q1              \n"                          \
  "vmax.f16   q4, q4, q3                \n"                          \
  "vmin.f16   q4, q4, q2                \n"                          \
  "vmul.f16   q10, q10, q4              \n"                          \
  "9:\n"

#define SPARSE_F16_F16_W4_v7_HARD_SWISH                              \
  /* do relu */                                                      \
  "12: \n"                                                           \
  "vld1.16   {d0-d3}, [%[hs_param]]!      @ load hard swish alpha\n" \
  "vmov.u16   d6,   #0                  @ for hardswish \n"          \
  "vld1.16   {d4-d5}, [%[hs_param]]    \n"                           \
  "vadd.f16   d8, d20,  d0               \n"                         \
  "vmul.f16   d20, d20, d2              \n"                          \
  "vmax.f16   d8, d8,   d6              \n"                          \
  "vmin.f16   d8, d8,   d4                \n"                        \
  "vmul.f16   d20, d20, d8              \n"                          \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_F16_F16_W48_v7_OUT                                  \
  SPARSE_F16_F16_W48_v7_KERNEL SPARSE_F16_F16_W48_v7_RELU          \
      SPARSE_F16_F16_W48_v7_RELU6 SPARSE_F16_F16_W48_v7_LEAKY_RELU \
          SPARSE_F16_F16_W48_v7_HARD_SWISH                         \
      "vst1.16   {d20-d23},  [%[c_ptr]]!\n"                        \
      "vst1.16   {d24-d27},  [%[c_ptr]]!\n"                        \
      "vst1.16   {d28-d31},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_F16_F16_W32_v7_OUT                                  \
  SPARSE_F16_F16_W32_v7_KERNEL SPARSE_F16_F16_W32_v7_RELU          \
      SPARSE_F16_F16_W32_v7_RELU6 SPARSE_F16_F16_W32_v7_LEAKY_RELU \
          SPARSE_F16_F16_W32_v7_HARD_SWISH                         \
      "vst1.16   {d20-d23},  [%[c_ptr]]!\n"                        \
      "vst1.16   {d24-d27},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_F16_F16_W16_v7_OUT                                  \
  SPARSE_F16_F16_W16_v7_KERNEL SPARSE_F16_F16_W16_v7_RELU          \
      SPARSE_F16_F16_W16_v7_RELU6 SPARSE_F16_F16_W16_v7_LEAKY_RELU \
          SPARSE_F16_F16_W16_v7_HARD_SWISH                         \
      "vst1.16   {d20-d23},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F16_F16_W8_v7_OUT                             \
  SPARSE_F16_F16_W8_v7_KERNEL SPARSE_F16_F16_W8_v7_RELU      \
      SPARSE_F16_F16_W8_v7_RELU6                             \
          SPARSE_F16_F16_W8_v7_LEAKY_RELU /* store result */ \
      SPARSE_F16_F16_W8_v7_HARD_SWISH "vst1.16   {d20-d21},  [%[c_ptr]]!\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_F16_F16_W4_v7_OUT                             \
  SPARSE_F16_F16_W4_v7_KERNEL SPARSE_F16_F16_W4_v7_RELU      \
      SPARSE_F16_F16_W4_v7_RELU6                             \
          SPARSE_F16_F16_W4_v7_LEAKY_RELU /* store result */ \
      SPARSE_F16_F16_W4_v7_HARD_SWISH "vst1.16   {d20},  [%[c_ptr]]!\n"

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
void sparse_conv_fp16_pipelined(const float16_t* A,
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
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  volatile float16_t alpha = 0.f;
  float16_t hs_param[24] = {0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3, hard_swish: 4
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = static_cast<float16_t>(act_param.Relu_clipped_coef);
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = static_cast<float16_t>(act_param.Leaky_relu_alpha);
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 0x04;
      for (int i = 0; i < 8; i++) {
        hs_param[i] = static_cast<float16_t>(act_param.hard_swish_offset);
        hs_param[i + 8] =
            static_cast<float16_t>(1.0 / act_param.hard_swish_scale);
        hs_param[i + 16] =
            static_cast<float16_t>(act_param.hard_swish_threshold);
      }
    }
  }
  int flag_bias = (bias != nullptr) ? 1 : 0;
  size_t mc = N * sizeof(float16_t);
  size_t nc = M;
  size_t output_stride = N * sizeof(float16_t);
  size_t output_decrement = output_stride * nc - 48 * sizeof(float16_t);
  while
    SPARSE_FP16_LIKELY(mc >= 48 * sizeof(float16_t)) {
      LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
        SPARSE_COMPUTE_LOOP
        float16_t* vhs_param = hs_param;
        // clang-format off
            asm volatile(SPARSE_F16_F16_W48_v7_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap),
                [hs_param] "+r"(vhs_param)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha)
              : "q0",
                "q1",
                "q2",
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
                "q14",
                "q15",
                "r0",
                "r2",
                "cc",
                "memory");
        // clang-format on
      }
      LITE_PARALLEL_COMMON_END();
      output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                            48 * sizeof(float16_t));
      B += 48;
      mc -= 48 * sizeof(float16_t);
    }

  if
    SPARSE_FP16_UNLIKELY(mc != 0) {
      if (mc >= (32 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          float16_t* vhs_param = hs_param;
          // clang-format off
            asm volatile(SPARSE_F16_F16_W32_v7_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap),
                [hs_param] "+r"(vhs_param)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha)
              : "q0",
                "q1",
                "q2",
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
                "q14",
                "q15",
                "r0",
                "r2",
                "cc",
                "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              32 * sizeof(float16_t));
        B += 32;
        mc -= 32 * sizeof(float16_t);
      }
      if (mc >= (16 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          float16_t* vhs_param = hs_param;
          // clang-format off
            asm volatile(SPARSE_F16_F16_W16_v7_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap),
                [hs_param] "+r"(vhs_param)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha)
              : "q0",
                "q1",
                "q2",
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
                "q14",
                "q15",
                "r0",
                "r2",
                "cc",
                "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              16 * sizeof(float16_t));
        B += 16;
        mc -= 16 * sizeof(float16_t);
      }
      if (mc >= (8 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          float16_t* vhs_param = hs_param;
          // clang-format off
            asm volatile(SPARSE_F16_F16_W8_v7_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap),
                [hs_param] "+r"(vhs_param)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha)
              : "q0",
                "q1",
                "q2",
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
                "q14",
                "q15",
                "r0",
                "r1",
                "cc",
                "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              8 * sizeof(float16_t));
        B += 8;
        mc -= 8 * sizeof(float16_t);
      }
      if (mc >= (4 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          float16_t* vhs_param = hs_param;
          // clang-format off
            asm volatile(SPARSE_F16_F16_W4_v7_OUT  
              : [a_ptr] "+r"(cur_w),
                [b_ptr] "+r"(cur_b),
                [c_ptr] "+r"(cur_output),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap),
                [hs_param] "+r"(vhs_param)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(alpha)
              : "q0",
                "q1",
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
                "q14",
                "q15",
                "r0",
                "r1",
                "cc",
                "memory");
          // clang-format on
        }
        LITE_PARALLEL_COMMON_END();
        output = reinterpret_cast<float16_t*>((uintptr_t)output +
                                              4 * sizeof(float16_t));
        B += 4;
        mc -= 4 * sizeof(float16_t);
      }
      while (mc >= (1 * sizeof(float16_t))) {
        LITE_PARALLEL_COMMON_BEGIN(i, tid, nc, 0, 1) {
          SPARSE_COMPUTE_LOOP
          float16_t vout0 = vbias;
          if
            SPARSE_FP16_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                vout0 += cur_b[0] * cur_w[0];
                cur_b = (const float16_t*)((uintptr_t)cur_b + (uintptr_t)diff);
                cur_w += 1;
              } while (--nnz != 0);
            }
          if (flag_act == 1) {
            vout0 = vout0 > 0.f ? vout0 : 0.f;
          } else if (flag_act == 0) {
          } else if (flag_act == 2) {
            vout0 = vout0 > 0.f ? vout0 : 0.f;
            vout0 = vout0 > alpha ? alpha : vout0;
          } else if (flag_act == 3) {
            vout0 = vout0 > 0.f ? vout0 : (vout0 * alpha);
          } else if (flag_act == 4) {
            auto tmp_0 =
                vout0 + static_cast<float16_t>(act_param.hard_swish_offset);
            auto tmp_1 =
                vout0 / static_cast<float16_t>(act_param.hard_swish_scale);
            tmp_0 = tmp_0 > 0.f ? (tmp_0 < static_cast<float16_t>(
                                               act_param.hard_swish_threshold)
                                       ? tmp_0
                                       : static_cast<float16_t>(
                                             act_param.hard_swish_threshold))
                                : 0.f;
            vout0 = tmp_0 * tmp_1;
          } else {
            LOG(FATAL) << "This act: " << flag_act << " doesn't support";
          }
          cur_output[0] = vout0;
        }
        LITE_PARALLEL_COMMON_END();
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
