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

#include "lite/backends/arm/math/sparse_conv_impl.h"
#include <arm_neon.h>
#include <vector>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#ifdef __aarch64__

#define SPARSE_F32_F32_W48_V8_KERNEL        \
  "dup     v20.4s,  %w[vbias]\n"            \
  "dup     v21.4s,  v20.s[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "dup     v22.4s,  v20.s[0]\n"             \
  "dup     v23.4s,  v20.s[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v24.4s,  v20.s[0]\n"             \
  "dup     v25.4s,  v20.s[0]\n"             \
  "dup     v26.4s,  v20.s[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "dup     v27.4s,  v20.s[0]\n"             \
  "dup     v28.4s,  v20.s[0]\n"             \
  "dup     v29.4s,  v20.s[0]\n"             \
  "dup     v30.4s,  v20.s[0]\n"             \
  "dup     v31.4s,  v20.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.4s,  v2.4s,  v0.s[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v21.4s,  v3.4s,  v0.s[0]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v22.4s,  v4.4s,  v0.s[0]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v23.4s,  v5.4s,  v0.s[0]\n"      \
  "fmla    v24.4s,  v6.4s,  v0.s[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v25.4s,  v7.4s,  v0.s[0]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v26.4s,  v8.4s,  v0.s[0]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v27.4s,  v9.4s,  v0.s[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v28.4s,  v10.4s,  v0.s[0]\n"     \
  "fmla    v29.4s,  v11.4s,  v0.s[0]\n"     \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v30.4s,  v12.4s,  v0.s[0]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[0]\n"     \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v20.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v21.4s,  v3.4s,  v0.s[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.4s,  v4.4s,  v0.s[1]\n"      \
  "fmla    v23.4s,  v5.4s,  v0.s[1]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v24.4s,  v6.4s,  v0.s[1]\n"      \
  "fmla    v25.4s,  v7.4s,  v0.s[1]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v26.4s,  v8.4s,  v0.s[1]\n"      \
  "fmla    v27.4s,  v9.4s,  v0.s[1]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v28.4s,  v10.4s,  v0.s[1]\n"     \
  "fmla    v29.4s,  v11.4s,  v0.s[1]\n"     \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v30.4s,  v12.4s,  v0.s[1]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[1]\n"     \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v20.4s,  v2.4s,  v0.s[2]\n"      \
  "fmla    v21.4s,  v3.4s,  v0.s[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v22.4s,  v4.4s,  v0.s[2]\n"      \
  "fmla    v23.4s,  v5.4s,  v0.s[2]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v24.4s,  v6.4s,  v0.s[2]\n"      \
  "fmla    v25.4s,  v7.4s,  v0.s[2]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v26.4s,  v8.4s,  v0.s[2]\n"      \
  "fmla    v27.4s,  v9.4s,  v0.s[2]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v28.4s,  v10.4s,  v0.s[2]\n"     \
  "fmla    v29.4s,  v11.4s,  v0.s[2]\n"     \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v30.4s,  v12.4s,  v0.s[2]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[2]\n"     \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "mov   w1, v1.s[3]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v20.4s,  v2.4s,  v0.s[3]\n"      \
  "fmla    v21.4s,  v3.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v22.4s,  v4.4s,  v0.s[3]\n"      \
  "fmla    v23.4s,  v5.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v24.4s,  v6.4s,  v0.s[3]\n"      \
  "fmla    v25.4s,  v7.4s,  v0.s[3]\n"      \
  "fmla    v26.4s,  v8.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v27.4s,  v9.4s,  v0.s[3]\n"      \
  "fmla    v28.4s,  v10.4s,  v0.s[3]\n"     \
  "fmla    v29.4s,  v11.4s,  v0.s[3]\n"     \
  "fmla    v30.4s,  v12.4s,  v0.s[3]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[3]\n"     \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.4s,  v2.4s,  v0.s[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v21.4s,  v3.4s,  v0.s[0]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v22.4s,  v4.4s,  v0.s[0]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v23.4s,  v5.4s,  v0.s[0]\n"      \
  "fmla    v24.4s,  v6.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v25.4s,  v7.4s,  v0.s[0]\n"      \
  "fmla    v26.4s,  v8.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v27.4s,  v9.4s,  v0.s[0]\n"      \
  "fmla    v28.4s,  v10.4s,  v0.s[0]\n"     \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v29.4s,  v11.4s,  v0.s[0]\n"     \
  "fmla    v30.4s,  v12.4s,  v0.s[0]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[0]\n"     \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v20.4s,  v2.4s,  v0.s[1]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v21.4s,  v3.4s,  v0.s[1]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.4s,  v4.4s,  v0.s[1]\n"      \
  "fmla    v23.4s,  v5.4s,  v0.s[1]\n"      \
  "fmla    v24.4s,  v6.4s,  v0.s[1]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v25.4s,  v7.4s,  v0.s[1]\n"      \
  "fmla    v26.4s,  v8.4s,  v0.s[1]\n"      \
  "fmla    v27.4s,  v9.4s,  v0.s[1]\n"      \
  "fmla    v28.4s,  v10.4s,  v0.s[1]\n"     \
  "fmla    v29.4s,  v11.4s,  v0.s[1]\n"     \
  "fmla    v30.4s,  v12.4s,  v0.s[1]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[1]\n"     \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v20.4s,  v2.4s,  v0.s[2]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "fmla    v21.4s,  v3.4s,  v0.s[2]\n"      \
  "ldp   q10, q11, [%[b_ptr], #128]\n"      \
  "fmla    v22.4s,  v4.4s,  v0.s[2]\n"      \
  "ldp   q12, q13, [%[b_ptr], #160]\n"      \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v23.4s,  v5.4s,  v0.s[2]\n"      \
  "fmla    v24.4s,  v6.4s,  v0.s[2]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #192]\n"     \
  "fmla    v25.4s,  v7.4s,  v0.s[2]\n"      \
  "fmla    v26.4s,  v8.4s,  v0.s[2]\n"      \
  "fmla    v27.4s,  v9.4s,  v0.s[2]\n"      \
  "fmla    v28.4s,  v10.4s,  v0.s[2]\n"     \
  "fmla    v29.4s,  v11.4s,  v0.s[2]\n"     \
  "fmla    v30.4s,  v12.4s,  v0.s[2]\n"     \
  "fmla    v31.4s,  v13.4s,  v0.s[2]\n"     \
  "1:\n"

#define SPARSE_F32_F32_W32_V8_KERNEL        \
  "dup     v21.4s,  %w[vbias]\n"            \
  "dup     v22.4s,  v21.s[0]\n"             \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "dup     v23.4s,  v21.s[0]\n"             \
  "dup     v24.4s,  v21.s[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v25.4s,  v21.s[0]\n"             \
  "dup     v26.4s,  v21.s[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "dup     v27.4s,  v21.s[0]\n"             \
  "dup     v28.4s,  v21.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.4s,  v4.4s,  v0.s[0]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[0]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v25.4s,  v6.4s,  v0.s[0]\n"      \
  "fmla    v26.4s,  v7.4s,  v0.s[0]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v27.4s,  v8.4s,  v0.s[0]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.4s,  v4.4s,  v0.s[1]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[1]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v25.4s,  v6.4s,  v0.s[1]\n"      \
  "fmla    v26.4s,  v7.4s,  v0.s[1]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v27.4s,  v8.4s,  v0.s[1]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[1]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.4s,  v4.4s,  v0.s[2]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[2]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "fmla    v25.4s,  v6.4s,  v0.s[2]\n"      \
  "fmla    v26.4s,  v7.4s,  v0.s[2]\n"      \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v27.4s,  v8.4s,  v0.s[2]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[2]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "mov   w1, v1.s[3]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[3]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v23.4s,  v4.4s,  v0.s[3]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[3]\n"      \
  "fmla    v25.4s,  v6.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v26.4s,  v7.4s,  v0.s[3]\n"      \
  "fmla    v27.4s,  v8.4s,  v0.s[3]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[3]\n"      \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v23.4s,  v4.4s,  v0.s[0]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v25.4s,  v6.4s,  v0.s[0]\n"      \
  "fmla    v26.4s,  v7.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v27.4s,  v8.4s,  v0.s[0]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[0]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[1]\n"      \
  "fmla    v23.4s,  v4.4s,  v0.s[1]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[1]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v25.4s,  v6.4s,  v0.s[1]\n"      \
  "fmla    v26.4s,  v7.4s,  v0.s[1]\n"      \
  "fmla    v27.4s,  v8.4s,  v0.s[1]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[1]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "ldp   q6, q7, [%[b_ptr], #64]\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "ldp   q8, q9, [%[b_ptr], #96]\n"         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v22.4s,  v3.4s,  v0.s[2]\n"      \
  "fmla    v23.4s,  v4.4s,  v0.s[2]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[2]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v25.4s,  v6.4s,  v0.s[2]\n"      \
  "fmla    v26.4s,  v7.4s,  v0.s[2]\n"      \
  "fmla    v27.4s,  v8.4s,  v0.s[2]\n"      \
  "fmla    v28.4s,  v9.4s,  v0.s[2]\n"      \
  "1:\n"

#define SPARSE_F32_F32_W16_V8_KERNEL        \
  "dup     v21.4s,  %w[vbias]\n"            \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "dup     v22.4s,  v21.s[0]\n"             \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "dup     v23.4s,  v21.s[0]\n"             \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "dup     v24.4s,  v21.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.4s,  v4.4s,  v0.s[0]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[0]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.4s,  v4.4s,  v0.s[1]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[1]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "fmla    v23.4s,  v4.4s,  v0.s[2]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[2]\n"      \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "mov   w1, v1.s[3]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v22.4s,  v3.4s,  v0.s[3]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v23.4s,  v4.4s,  v0.s[3]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[3]\n"      \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "sxtw  x1,  w1\n"                         \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v23.4s,  v4.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v24.4s,  v5.4s,  v0.s[0]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[1]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v23.4s,  v4.4s,  v0.s[1]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[1]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "ldp   q4, q5, [%[b_ptr], #32]\n"         \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[2]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v23.4s,  v4.4s,  v0.s[2]\n"      \
  "fmla    v24.4s,  v5.4s,  v0.s[2]\n"      \
  "1:\n"

#define SPARSE_F32_F32_W8_V8_KERNEL         \
  "dup     v21.4s,  %w[vbias]\n"            \
  "dup     v22.4s,  v21.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "subs    %w[n],   %w[n],   #1\n"          \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[1]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[2]\n"      \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[3]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[3]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[3]\n"      \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "subs  %w[m],   %w[m],   #1\n"            \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "subs  %w[m],   %w[m],   #1\n"            \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[1]\n"      \
  "beq     1f\n"                            \
  "ldp   q2, q3, [%[b_ptr]]\n"              \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "fmla    v22.4s,  v3.4s,  v0.s[2]\n"      \
  "1:\n"

#define SPARSE_F32_F32_W4_V8_KERNEL         \
  "dup     v21.4s,  %w[vbias]\n"            \
  "cbz    %w[k],    1f\n"                   \
  "cbz    %w[n],    3f\n"                   \
  "0:\n"                                    \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "mov   w1, v1.s[0]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr]]\n"           \
  "subs    %w[n],   %w[n],   #1\n"          \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr]]\n"           \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr]]\n"           \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[3]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "fmla    v21.4s,  v2.4s,  v0.s[3]\n"      \
  "bne     0b\n"                            \
  "3:\n"                                    \
  "cbz    %w[m],    1f\n"                   \
  "ldr   q0, [%[a_ptr]], #16\n"             \
  "ldr   q1, [%[widx_dmap]],   #16\n"       \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "subs  %w[m],   %w[m],   #1\n"            \
  "mov   w1, v1.s[0]\n"                     \
  "prfm  pldl1keep, [%[widx_dmap], #128]\n" \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[a_ptr], #128]\n"     \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"      \
  "beq     1f\n"                            \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[1]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr]]\n"           \
  "subs  %w[m],   %w[m],   #1\n"            \
  "fmla    v21.4s,  v2.4s,  v0.s[1]\n"      \
  "beq     1f\n"                            \
  "ldr   q2, [%[b_ptr]]\n"                  \
  "mov   w1, v1.s[2]\n"                     \
  "sxtw  x1,  w1\n"                         \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "prfm  pldl1keep, [%[b_ptr]]\n"           \
  "fmla    v21.4s,  v2.4s,  v0.s[2]\n"      \
  "1:\n"

#define SPARSE_F32_F32_W48_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v0.4s, #0\n"               \
  "fmax   v20.4s, v20.4s, v0.4s\n"   \
  "fmax   v21.4s, v21.4s, v0.4s\n"   \
  "fmax   v22.4s, v22.4s, v0.4s\n"   \
  "fmax   v23.4s, v23.4s, v0.4s\n"   \
  "fmax   v24.4s, v24.4s, v0.4s\n"   \
  "fmax   v25.4s, v25.4s, v0.4s\n"   \
  "fmax   v26.4s, v26.4s, v0.4s\n"   \
  "fmax   v27.4s, v27.4s, v0.4s\n"   \
  "fmax   v28.4s, v28.4s, v0.4s\n"   \
  "fmax   v29.4s, v29.4s, v0.4s\n"   \
  "fmax   v30.4s, v30.4s, v0.4s\n"   \
  "fmax   v31.4s, v31.4s, v0.4s\n"   \
  "b      9f                    \n"

#define SPARSE_F32_F32_W32_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v30.4s, #0\n"              \
  "fmax   v21.4s, v21.4s, v30.4s\n"  \
  "fmax   v22.4s, v22.4s, v30.4s\n"  \
  "fmax   v23.4s, v23.4s, v30.4s\n"  \
  "fmax   v24.4s, v24.4s, v30.4s\n"  \
  "fmax   v25.4s, v25.4s, v30.4s\n"  \
  "fmax   v26.4s, v26.4s, v30.4s\n"  \
  "fmax   v27.4s, v27.4s, v30.4s\n"  \
  "fmax   v28.4s, v28.4s, v30.4s\n"  \
  "b      9f                    \n"

#define SPARSE_F32_F32_W16_V8_RELU   \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.4s, #0\n"               \
  "fmax   v21.4s, v21.4s, v9.4s\n"   \
  "fmax   v22.4s, v22.4s, v9.4s\n"   \
  "fmax   v23.4s, v23.4s, v9.4s\n"   \
  "fmax   v24.4s, v24.4s, v9.4s\n"   \
  "b      9f                    \n"

#define SPARSE_F32_F32_W8_V8_RELU    \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.4s, #0\n"               \
  "fmax   v21.4s, v21.4s, v9.4s\n"   \
  "fmax   v22.4s, v22.4s, v9.4s\n"   \
  "b      9f                    \n"

#define SPARSE_F32_F32_W4_V8_RELU    \
  /* do relu */                      \
  "cmp    %w[vflag_act],    #0\n"    \
  "beq   9f                     \n"  \
  "cmp    %w[vflag_act],    #1\n"    \
  "bne   10f                     \n" \
  "movi   v9.4s, #0\n"               \
  "fmax   v21.4s, v21.4s, v9.4s\n"   \
  "b      9f                    \n"

#define SPARSE_F32_F32_W48_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.4s, #0\n"                \
  "dup    v1.4s,  %w[valpha]\n"       \
  "fmax   v20.4s, v20.4s, v0.4s\n"    \
  "fmax   v21.4s, v21.4s, v0.4s\n"    \
  "fmax   v22.4s, v22.4s, v0.4s\n"    \
  "fmax   v23.4s, v23.4s, v0.4s\n"    \
  "fmax   v24.4s, v24.4s, v0.4s\n"    \
  "fmax   v25.4s, v25.4s, v0.4s\n"    \
  "fmax   v26.4s, v26.4s, v0.4s\n"    \
  "fmax   v27.4s, v27.4s, v0.4s\n"    \
  "fmax   v28.4s, v28.4s, v0.4s\n"    \
  "fmax   v29.4s, v29.4s, v0.4s\n"    \
  "fmax   v30.4s, v30.4s, v0.4s\n"    \
  "fmax   v31.4s, v31.4s, v0.4s\n"    \
  "fmin   v20.4s, v20.4s, v1.4s\n"    \
  "fmin   v21.4s, v21.4s, v1.4s\n"    \
  "fmin   v22.4s, v22.4s, v1.4s\n"    \
  "fmin   v23.4s, v23.4s, v1.4s\n"    \
  "fmin   v24.4s, v24.4s, v1.4s\n"    \
  "fmin   v25.4s, v25.4s, v1.4s\n"    \
  "fmin   v26.4s, v26.4s, v1.4s\n"    \
  "fmin   v27.4s, v27.4s, v1.4s\n"    \
  "fmin   v28.4s, v28.4s, v1.4s\n"    \
  "fmin   v29.4s, v29.4s, v1.4s\n"    \
  "fmin   v30.4s, v30.4s, v1.4s\n"    \
  "fmin   v31.4s, v31.4s, v1.4s\n"    \
  "b      9f                    \n"

#define SPARSE_F32_F32_W32_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.4s, #0\n"                \
  "dup    v1.4s,  %w[valpha]\n"       \
  "fmax   v21.4s, v21.4s, v0.4s\n"    \
  "fmax   v22.4s, v22.4s, v0.4s\n"    \
  "fmax   v23.4s, v23.4s, v0.4s\n"    \
  "fmax   v24.4s, v24.4s, v0.4s\n"    \
  "fmax   v25.4s, v25.4s, v0.4s\n"    \
  "fmax   v26.4s, v26.4s, v0.4s\n"    \
  "fmax   v27.4s, v27.4s, v0.4s\n"    \
  "fmax   v28.4s, v28.4s, v0.4s\n"    \
  "fmin   v21.4s, v21.4s, v1.4s\n"    \
  "fmin   v22.4s, v22.4s, v1.4s\n"    \
  "fmin   v23.4s, v23.4s, v1.4s\n"    \
  "fmin   v24.4s, v24.4s, v1.4s\n"    \
  "fmin   v25.4s, v25.4s, v1.4s\n"    \
  "fmin   v26.4s, v26.4s, v1.4s\n"    \
  "fmin   v27.4s, v27.4s, v1.4s\n"    \
  "fmin   v28.4s, v28.4s, v1.4s\n"    \
  "b      9f                    \n"

#define SPARSE_F32_F32_W16_V8_RELU6   \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.4s, #0\n"                \
  "dup    v1.4s,  %w[valpha]\n"       \
  "fmax   v21.4s, v21.4s, v0.4s\n"    \
  "fmax   v22.4s, v22.4s, v0.4s\n"    \
  "fmax   v23.4s, v23.4s, v0.4s\n"    \
  "fmax   v24.4s, v24.4s, v0.4s\n"    \
  "fmin   v21.4s, v21.4s, v1.4s\n"    \
  "fmin   v22.4s, v22.4s, v1.4s\n"    \
  "fmin   v23.4s, v23.4s, v1.4s\n"    \
  "fmin   v24.4s, v24.4s, v1.4s\n"    \
  "b      9f                    \n"

#define SPARSE_F32_F32_W8_V8_RELU6    \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.4s, #0\n"                \
  "dup    v1.4s,  %w[valpha]\n"       \
  "fmax   v21.4s, v21.4s, v0.4s\n"    \
  "fmax   v22.4s, v22.4s, v0.4s\n"    \
  "fmin   v21.4s, v21.4s, v1.4s\n"    \
  "fmin   v22.4s, v22.4s, v1.4s\n"    \
  "b      9f                    \n"

#define SPARSE_F32_F32_W4_V8_RELU6    \
  /* do relu6 */                      \
  "10: \n"                            \
  "cmp   %w[vflag_act],  #2       \n" \
  "bne   11f                     \n"  \
  "movi   v0.4s, #0\n"                \
  "dup    v1.4s,  %w[valpha]\n"       \
  "fmax   v21.4s, v21.4s, v0.4s\n"    \
  "fmin   v21.4s, v21.4s, v1.4s\n"    \
  "b      9f                    \n"

#define SPARSE_F32_F32_W48_V8_LEAKY_RELU                            \
  /* do relu */                                                     \
  "11: \n"                                                          \
  "movi   v0.4s, #0\n"                      /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"             /* leakey relu alpha */ \
  "fcmge  v2.4s,    v20.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v3.4s,    v20.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v4.4s,    v21.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v5.4s,    v21.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v6.4s,    v22.4s,   v0.4s   \n"   /* vcgeq_f32 */         \
  "fmul   v7.4s,    v22.4s,   v1.4s   \n"   /* vmulq_f32 */         \
  "fcmge  v8.4s,    v23.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v9.4s,    v23.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v10.4s,   v24.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v11.4s,   v24.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v12.4s,   v25.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v13.4s,   v25.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "bif    v20.16b,  v3.16b,   v2.16b  \n"   /* choose*/             \
  "bif    v21.16b,  v5.16b,   v4.16b  \n"   /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v24.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v25.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "fcmge  v2.4s,    v26.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v3.4s,    v26.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v4.4s,    v27.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v5.4s,    v27.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v6.4s,    v28.4s,   v0.4s   \n"   /* vcgeq_f32 */         \
  "fmul   v7.4s,    v28.4s,   v1.4s   \n"   /* vmulq_f32 */         \
  "fcmge  v8.4s,    v29.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v9.4s,    v29.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v10.4s,   v30.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v11.4s,   v30.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v12.4s,   v31.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v13.4s,   v31.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "bif    v26.16b,  v3.16b,   v2.16b  \n"   /* choose*/             \
  "bif    v27.16b,  v5.16b,   v4.16b  \n"   /* choose*/             \
  "bif    v28.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v29.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v30.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v31.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "9:\n"

#define SPARSE_F32_F32_W32_V8_LEAKY_RELU                           \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v23.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v23.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v23.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v24.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "fcmge  v2.4s,    v25.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v25.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v26.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v26.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v27.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v27.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v28.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v28.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v25.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v26.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v27.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v28.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "9:\n"

#define SPARSE_F32_F32_W16_V8_LEAKY_RELU                           \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v23.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v23.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v23.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v24.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "9:\n"

#define SPARSE_F32_F32_W8_V8_LEAKY_RELU                            \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "9:\n"

#define SPARSE_F32_F32_W4_V8_LEAKY_RELU                            \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_F32_F32_W48_V8_OUT       \
  SPARSE_F32_F32_W48_V8_KERNEL          \
  SPARSE_F32_F32_W48_V8_RELU            \
  SPARSE_F32_F32_W48_V8_RELU6           \
  SPARSE_F32_F32_W48_V8_LEAKY_RELU      \
  /* store result */                    \
  "stp   q20, q21,  [%[c_ptr]]\n"       \
  "stp   q22, q23,  [%[c_ptr], #32]\n"  \
  "stp   q24, q25,  [%[c_ptr], #64]\n"  \
  "stp   q26, q27,  [%[c_ptr], #96]\n"  \
  "stp   q28, q29,  [%[c_ptr], #128]\n" \
  "stp   q30, q31,  [%[c_ptr], #160]\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx32.
 */
#define SPARSE_F32_F32_W32_V8_OUT      \
  SPARSE_F32_F32_W32_V8_KERNEL         \
  SPARSE_F32_F32_W32_V8_RELU           \
  SPARSE_F32_F32_W32_V8_RELU6          \
  SPARSE_F32_F32_W32_V8_LEAKY_RELU     \
  /* store result */                   \
  "stp   q21, q22,  [%[c_ptr]]\n"      \
  "stp   q23, q24,  [%[c_ptr], #32]\n" \
  "stp   q25, q26,  [%[c_ptr], #64]\n" \
  "stp   q27, q28,  [%[c_ptr], #96]\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_F32_F32_W16_V8_OUT  \
  SPARSE_F32_F32_W16_V8_KERNEL     \
  SPARSE_F32_F32_W16_V8_RELU       \
  SPARSE_F32_F32_W16_V8_RELU6      \
  SPARSE_F32_F32_W16_V8_LEAKY_RELU \
  /* store result */               \
  "stp   q21, q22,  [%[c_ptr]]\n"  \
  "stp   q23, q24,  [%[c_ptr], #32]\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F32_F32_W8_V8_OUT  \
  SPARSE_F32_F32_W8_V8_KERNEL     \
  SPARSE_F32_F32_W8_V8_RELU       \
  SPARSE_F32_F32_W8_V8_RELU6      \
  SPARSE_F32_F32_W8_V8_LEAKY_RELU \
  /* store result */              \
  "stp   q21, q22,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_F32_F32_W4_V8_OUT  \
  SPARSE_F32_F32_W4_V8_KERNEL     \
  SPARSE_F32_F32_W4_V8_RELU       \
  SPARSE_F32_F32_W4_V8_RELU6      \
  SPARSE_F32_F32_W4_V8_LEAKY_RELU \
  /* store result */              \
  "str   q21,  [%[c_ptr]]\n"

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are f32.
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
void sparse_conv_fp32_pipelined(const float* A,
                                const float* B,
                                const int32_t* widx_dmap,
                                const uint32_t* nidx_nnzmap,
                                const float* bias,
                                float* output,
                                const int M,
                                const int K,
                                const int N,
                                const operators::SparseConvParam& param,
                                ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  int flag_bias = (bias != nullptr) ? 1 : 0;
  size_t mc = N * sizeof(float);
  size_t nc = M;
  size_t output_stride = N * sizeof(float);
  size_t output_decrement = output_stride * nc - 48 * sizeof(float);
  while
    SPARSE_LIKELY(mc >= 48 * sizeof(float)) {
      const float* w = A;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      float valpha = alpha;

      for (size_t i = 0; i < nc; i++) {
        uint32_t nnz = *nnzmap++;
        uint32_t pair_num = nnz / 4;
        uint32_t lave_num = nnz % 4;
        float vbias = (bias != nullptr) ? bias[i] : 0.0;
        // clang-format off
            asm volatile(SPARSE_F32_F32_W48_V8_OUT  
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w1", "x1", "cc", "memory");
        // clang-format on
        output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
      }
      output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
      B += 48;
      mc -= 48 * sizeof(float);
    }

  if
    SPARSE_UNLIKELY(mc != 0) {
      output_decrement += 16 * sizeof(float);
      if (mc & (32 * sizeof(float))) {
        const float* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          uint32_t pair_num = nnz / 4;
          uint32_t lave_num = nnz % 4;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W32_V8_OUT  
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 32;
        mc -= 32 * sizeof(float);
      }
      output_decrement += 16 * sizeof(float);
      if (mc & (16 * sizeof(float))) {
        const float* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          uint32_t pair_num = nnz / 4;
          uint32_t lave_num = nnz % 4;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W16_V8_OUT  
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v11", "v12", "v13", "v14", "v21", "v22", "v23",
                "v24", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 16;
        mc -= 16 * sizeof(float);
      }
      output_decrement += 8 * sizeof(float);
      if (mc & (8 * sizeof(float))) {
        const float* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          uint32_t pair_num = nnz / 4;
          uint32_t lave_num = nnz % 4;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W8_V8_OUT  
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v9", "v11", "v12", "v21", 
              "v22", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 8;
        mc -= 8 * sizeof(float);
      }
      output_decrement += 4 * sizeof(float);
      if (mc & (4 * sizeof(float))) {
        const float* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          uint32_t pair_num = nnz / 4;
          uint32_t lave_num = nnz % 4;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W4_V8_OUT  
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [n] "+r"(pair_num),
                [m] "+r"(lave_num),
                [widx_dmap] "+r"(dmap)
              : [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v9", "v11", "v21", 
              "w1", "w2", "w3", "w4", "w5", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 4;
        mc -= 4 * sizeof(float);
      }

      if
        SPARSE_UNLIKELY(mc != 0 && mc < 4 * sizeof(float)) {
          const float* w = A;
          const int32_t* dmap = widx_dmap;
          const uint32_t* nnzmap = nidx_nnzmap;
          const float* bs = bias;
          float val = alpha;
          int mindex = mc / sizeof(float);

          for (size_t i = 0; i < nc; i++) {
            float vbias = (bias != nullptr) ? *bs++ : 0;
            for (size_t k = 0; k < mindex; k++) {
              *(output + k) = vbias;
            }
            uint32_t nnz = *nnzmap++;
            for (size_t j = 0; j < nnz; j++) {
              for (size_t k = 0; k < mindex; k++) {
                *(output + k) += (*w) * (*(B + k));
              }
              w += 1;
              intptr_t diff = *dmap++;
              B = (const float*)((uintptr_t)B + (uintptr_t)diff);
            }
            size_t re = nnz % 4;
            if (re != 0) {
              for (int j = 0; j < (4 - re); j++) {
                w++;
                dmap++;
              }
            }
            switch (flag_act) {
              case 0:
                break;
              case 1:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) = *(output + k) > 0 ? *(output + k) : 0;
                }
                break;
              case 2:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) = *(output + k) > 0 ? *(output + k) : 0;
                  *(output + k) = *(output + k) < val ? *(output + k) : val;
                }
                break;
              default:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) =
                      *(output + k) >= 0 ? *(output + k) : *(output + k) * val;
                }
                break;
            }
            output =
                reinterpret_cast<float*>((uintptr_t)output + output_stride);
          }
        }
    }
}

#define SPARSE_INT8_F32_W48_V8_KERNEL            \
  "eor v8.16b, v0.16b, v0.16b\n"                 \
  "eor v9.16b, v1.16b, v1.16b\n"                 \
  "eor v10.16b, v2.16b, v2.16b\n"                \
  "eor v11.16b, v3.16b, v3.16b\n"                \
  "eor v12.16b, v4.16b, v4.16b\n"                \
  "prfm  pldl1keep, [%[a_ptr], #32]\n"           \
  "eor v13.16b, v5.16b, v5.16b\n"                \
  "eor v14.16b, v6.16b, v6.16b\n"                \
  "prfm  pldl1keep, [%[widx_dmap], #32]\n"       \
  "eor v15.16b, v7.16b, v7.16b\n"                \
  "eor v16.16b, v0.16b, v0.16b\n"                \
  "prfm  pldl1keep, [%[b_ptr], #48]\n"           \
  "eor v17.16b, v1.16b, v1.16b\n"                \
  "eor v18.16b, v2.16b, v2.16b\n"                \
  "eor v19.16b, v3.16b, v3.16b\n"                \
  "dup     v20.4s,  %w[vbias]\n"                 \
  "dup     v21.4s,  v20.s[0]\n"                  \
  "dup     v22.4s,  v20.s[0]\n"                  \
  "dup     v23.4s,  v20.s[0]\n"                  \
  "dup     v24.4s,  v20.s[0]\n"                  \
  "dup     v25.4s,  v20.s[0]\n"                  \
  "dup     v26.4s,  v20.s[0]\n"                  \
  "dup     v27.4s,  v20.s[0]\n"                  \
  "dup     v28.4s,  v20.s[0]\n"                  \
  "dup     v29.4s,  v20.s[0]\n"                  \
  "dup     v30.4s,  v20.s[0]\n"                  \
  "dup     v31.4s,  v20.s[0]\n"                  \
  "cbz    %w[k],    1f\n"                        \
  "0:\n"                                         \
  "ld1r  {v0.16b}, [%[a_ptr]], #1\n"             \
  "ldr   w1, [%[widx_dmap]],   #4\n"             \
  "sxtw  x1,  w1\n"                              \
  "ld1   {v1.16b, v2.16b, v3.16b}, [%[b_ptr]]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"              \
  "smull   v4.8h,   v0.8b,   v1.8b\n"            \
  "smull2  v5.8h,   v0.16b,  v1.16b\n"           \
  "smull   v6.8h,   v0.8b,   v2.8b\n"            \
  "smull2  v7.8h,   v0.16b,  v2.16b\n"           \
  "subs    %w[k],   %w[k],   #1\n"               \
  "saddw   v8.4s,  v8.4s,  v4.4h\n"              \
  "saddw2  v9.4s,  v9.4s,  v4.8h\n"              \
  "prfm  pldl1keep, [%[b_ptr], #48]\n"           \
  "saddw   v10.4s,  v10.4s,  v5.4h\n"            \
  "saddw2  v11.4s,  v11.4s,  v5.8h\n"            \
  "saddw   v12.4s,  v12.4s,  v6.4h\n"            \
  "saddw2  v13.4s,  v13.4s,  v6.8h\n"            \
  "saddw   v14.4s,  v14.4s,  v7.4h\n"            \
  "saddw2  v15.4s,  v15.4s,  v7.8h\n"            \
  "smull   v4.8h,   v0.8b,   v3.8b\n"            \
  "smull2  v5.8h,   v0.16b,  v3.16b\n"           \
  "saddw   v16.4s,  v16.4s,  v4.4h\n"            \
  "saddw2  v17.4s,  v17.4s,  v4.8h\n"            \
  "saddw   v18.4s,  v18.4s,  v5.4h\n"            \
  "saddw2  v19.4s,  v19.4s,  v5.8h\n"            \
  "bne     0b\n"                                 \
  "1:\n"                                         \
  "dup     v0.4s,  %w[vscale]\n"                 \
  "scvtf   v1.4s,  v8.4s\n"                      \
  "scvtf   v2.4s,  v9.4s\n"                      \
  "scvtf   v3.4s,  v10.4s\n"                     \
  "scvtf   v4.4s,  v11.4s\n"                     \
  "scvtf   v5.4s,  v12.4s\n"                     \
  "scvtf   v6.4s,  v13.4s\n"                     \
  "scvtf   v7.4s,  v14.4s\n" /* scale */         \
  "fmla    v20.4s,  v1.4s,  v0.s[0]\n"           \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"           \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"           \
  "fmla    v23.4s,  v4.4s,  v0.s[0]\n"           \
  "fmla    v24.4s,  v5.4s,  v0.s[0]\n"           \
  "fmla    v25.4s,  v6.4s,  v0.s[0]\n"           \
  "fmla    v26.4s,  v7.4s,  v0.s[0]\n"           \
  "scvtf   v1.4s,  v15.4s\n"                     \
  "scvtf   v2.4s,  v16.4s\n"                     \
  "scvtf   v3.4s,  v17.4s\n"                     \
  "scvtf   v4.4s,  v18.4s\n"                     \
  "scvtf   v5.4s,  v19.4s\n" /* scale */         \
  "fmla    v27.4s,  v1.4s,  v0.s[0]\n"           \
  "fmla    v28.4s,  v2.4s,  v0.s[0]\n"           \
  "fmla    v29.4s,  v3.4s,  v0.s[0]\n"           \
  "fmla    v30.4s,  v4.4s,  v0.s[0]\n"           \
  "fmla    v31.4s,  v5.4s,  v0.s[0]\n"

#define SPARSE_INT8_F32_W32_V8_KERNEL       \
  "eor v11.16b, v0.16b, v0.16b\n"           \
  "eor v12.16b, v1.16b, v1.16b\n"           \
  "prfm  pldl1keep, [%[a_ptr], #32]\n"      \
  "eor v13.16b, v2.16b, v2.16b\n"           \
  "eor v14.16b, v3.16b, v3.16b\n"           \
  "prfm  pldl1keep, [%[widx_dmap], #32]\n"  \
  "eor v15.16b, v4.16b, v4.16b\n"           \
  "eor v16.16b, v5.16b, v5.16b\n"           \
  "prfm  pldl1keep, [%[b_ptr], #32]\n"      \
  "eor v17.16b, v6.16b, v6.16b\n"           \
  "eor v18.16b, v7.16b, v7.16b\n"           \
  "dup     v21.4s,  %w[vbias]\n"            \
  "dup     v22.4s,  v21.s[0]\n"             \
  "dup     v23.4s,  v21.s[0]\n"             \
  "dup     v24.4s,  v21.s[0]\n"             \
  "dup     v25.4s,  v21.s[0]\n"             \
  "dup     v26.4s,  v21.s[0]\n"             \
  "dup     v27.4s,  v21.s[0]\n"             \
  "dup     v28.4s,  v21.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "0:\n"                                    \
  "ld1r  {v0.16b}, [%[a_ptr]], #1\n"        \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "sxtw  x1,  w1\n"                         \
  "ld1   {v1.16b, v2.16b}, [%[b_ptr]]\n"    \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "smull   v3.8h,   v0.8b,   v1.8b\n"       \
  "smull2  v4.8h,   v0.16b,  v1.16b\n"      \
  "subs    %w[k],   %w[k],   #1\n"          \
  "smull   v5.8h,   v0.8b,   v2.8b\n"       \
  "smull2  v6.8h,   v0.16b,  v2.16b\n"      \
  "prfm  pldl1keep, [%[b_ptr], #32]\n"      \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"       \
  "saddw2  v12.4s,  v12.4s,  v3.8h\n"       \
  "saddw   v13.4s,  v13.4s,  v4.4h\n"       \
  "saddw2  v14.4s,  v14.4s,  v4.8h\n"       \
  "saddw   v15.4s,  v15.4s,  v5.4h\n"       \
  "saddw2  v16.4s,  v16.4s,  v5.8h\n"       \
  "saddw   v17.4s,  v17.4s,  v6.4h\n"       \
  "saddw2  v18.4s,  v18.4s,  v6.8h\n"       \
  "bne     0b\n"                            \
  "scvtf   v3.4s,  v11.4s\n"                \
  "scvtf   v4.4s,  v12.4s\n"                \
  "scvtf   v5.4s,  v13.4s\n"                \
  "scvtf   v6.4s,  v14.4s\n"                \
  "scvtf   v7.4s,  v15.4s\n"                \
  "scvtf   v8.4s,  v16.4s\n"                \
  "scvtf   v9.4s,  v17.4s\n"                \
  "scvtf   v10.4s, v18.4s\n" /* add bias */ \
  "dup     v31.4s,  %w[vscale]\n"           \
  "fmla    v21.4s,  v3.4s,  v31.s[0]\n"     \
  "fmla    v22.4s,  v4.4s,  v31.s[0]\n"     \
  "fmla    v23.4s,  v5.4s,  v31.s[0]\n"     \
  "fmla    v24.4s,  v6.4s,  v31.s[0]\n"     \
  "fmla    v25.4s,  v7.4s,  v31.s[0]\n"     \
  "fmla    v26.4s,  v8.4s,  v31.s[0]\n"     \
  "fmla    v27.4s,  v9.4s,  v31.s[0]\n"     \
  "fmla    v28.4s,  v10.4s, v31.s[0]\n"     \
  "1:\n"

#define SPARSE_INT8_F32_W16_V8_KERNEL       \
  "eor v11.16b, v0.16b, v0.16b\n"           \
  "eor v12.16b, v1.16b, v1.16b\n"           \
  "eor v13.16b, v2.16b, v2.16b\n"           \
  "eor v14.16b, v3.16b, v3.16b\n"           \
  "prfm  pldl1keep, [%[b_ptr], #16]\n"      \
  "dup     v21.4s,  %w[vbias]\n"            \
  "dup     v22.4s,  v21.s[0]\n"             \
  "dup     v23.4s,  v21.s[0]\n"             \
  "dup     v24.4s,  v21.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "0:\n"                                    \
  "ld1r  {v0.16b}, [%[a_ptr]], #1\n"        \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "sxtw  x1,  w1\n"                         \
  "ld1   {v1.16b}, [%[b_ptr]]\n"            \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "smull   v3.8h,   v0.8b,   v1.8b\n"       \
  "smull2  v4.8h,   v0.16b,  v1.16b\n"      \
  "prfm  pldl1keep, [%[b_ptr], #16]\n"      \
  "subs    %w[k],   %w[k],   #1\n"          \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"       \
  "saddw2  v12.4s,  v12.4s,  v3.8h\n"       \
  "saddw   v13.4s,  v13.4s,  v4.4h\n"       \
  "saddw2  v14.4s,  v14.4s,  v4.8h\n"       \
  "bne     0b\n"                            \
  "scvtf   v5.4s,  v11.4s\n"                \
  "scvtf   v6.4s,  v12.4s\n"                \
  "scvtf   v7.4s,  v13.4s\n"                \
  "scvtf   v8.4s,  v14.4s\n" /* add bias */ \
  "dup     v2.4s,  %w[vscale]\n"            \
  "fmla    v21.4s,  v5.4s,  v2.s[0]\n"      \
  "fmla    v22.4s,  v6.4s,  v2.s[0]\n"      \
  "fmla    v23.4s,  v7.4s,  v2.s[0]\n"      \
  "fmla    v24.4s,  v8.4s,  v2.s[0]\n"      \
  "1:\n"

#define SPARSE_INT8_F32_W8_V8_KERNEL        \
  "eor v11.16b, v0.16b, v0.16b\n"           \
  "eor v12.16b, v1.16b, v1.16b\n"           \
  "dup     v21.4s,  %w[vbias]\n"            \
  "dup     v22.4s,  v21.s[0]\n"             \
  "cbz    %w[k],    1f\n"                   \
  "0:\n"                                    \
  "ld1r  {v0.8b}, [%[a_ptr]], #1\n"         \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "sxtw  x1,  w1\n"                         \
  "ld1   {v1.8b}, [%[b_ptr]]\n"             \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "smull   v3.8h,   v0.8b,   v1.8b\n"       \
  "subs    %w[k],   %w[k],   #1\n"          \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"       \
  "saddw2  v12.4s,  v12.4s,  v3.8h\n"       \
  "bne     0b\n"                            \
  "scvtf   v4.4s,  v11.4s\n"                \
  "scvtf   v5.4s,  v12.4s\n" /* add bias */ \
  "dup     v2.4s,   %w[vscale]\n"           \
  "fmla    v21.4s,  v4.4s,  v2.s[0]\n"      \
  "fmla    v22.4s,  v5.4s,  v2.s[0]\n"      \
  "1:\n"

#define SPARSE_INT8_F32_W4_V8_KERNEL        \
  "eor v11.16b, v0.16b, v0.16b\n"           \
  "dup     v21.4s,  %w[vbias]\n"            \
  "cbz    %w[k],    1f\n"                   \
  "0:\n"                                    \
  "ld1r  {v0.8b}, [%[a_ptr]], #1\n"         \
  "ldr   w1, [%[widx_dmap]],   #4\n"        \
  "ldrsb   w2, [%[b_ptr]]\n"                \
  "ldrsb   w3, [%[b_ptr], #1]\n"            \
  "ldrsb   w4, [%[b_ptr], #2]\n"            \
  "ldrsb   w5, [%[b_ptr], #3]\n"            \
  "sxtw  x1,  w1\n"                         \
  "mov   v1.b[0], w2\n"                     \
  "mov   v1.b[1], w3\n"                     \
  "mov   v1.b[2], w4\n"                     \
  "mov   v1.b[3], w5\n"                     \
  "add   %[b_ptr],  %[b_ptr], x1\n"         \
  "smull   v3.8h,   v0.8b,   v1.8b\n"       \
  "subs    %w[k],   %w[k],   #1\n"          \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"       \
  "bne     0b\n"                            \
  "scvtf   v4.4s,  v11.4s\n" /* add bias */ \
  "dup     v2.4s,   %w[vscale]\n"           \
  "fmla    v21.4s,  v4.4s,  v2.s[0]\n"      \
  "1:\n"

#define SPARSE_INT8_F32_W48_V8_RELU                   \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v0.4s, #0\n"               /* for relu */   \
  "fmax   v20.4s, v20.4s, v0.4s\n"   /* relu */       \
  "fmax   v21.4s, v21.4s, v0.4s\n"   /* relu */       \
  "fmax   v22.4s, v22.4s, v0.4s\n"   /* relu */       \
  "fmax   v23.4s, v23.4s, v0.4s\n"   /* relu */       \
  "fmax   v24.4s, v24.4s, v0.4s\n"   /* relu */       \
  "fmax   v25.4s, v25.4s, v0.4s\n"   /* relu */       \
  "fmax   v26.4s, v26.4s, v0.4s\n"   /* relu */       \
  "fmax   v27.4s, v27.4s, v0.4s\n"   /* relu */       \
  "fmax   v28.4s, v28.4s, v0.4s\n"   /* relu */       \
  "fmax   v29.4s, v29.4s, v0.4s\n"   /* relu */       \
  "fmax   v30.4s, v30.4s, v0.4s\n"   /* relu */       \
  "fmax   v31.4s, v31.4s, v0.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W32_V8_RELU                   \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v30.4s, #0\n"              /* for relu */   \
  "fmax   v21.4s, v21.4s, v30.4s\n"  /* relu */       \
  "fmax   v22.4s, v22.4s, v30.4s\n"  /* relu */       \
  "fmax   v23.4s, v23.4s, v30.4s\n"  /* relu */       \
  "fmax   v24.4s, v24.4s, v30.4s\n"  /* relu */       \
  "fmax   v25.4s, v25.4s, v30.4s\n"  /* relu */       \
  "fmax   v26.4s, v26.4s, v30.4s\n"  /* relu */       \
  "fmax   v27.4s, v27.4s, v30.4s\n"  /* relu */       \
  "fmax   v28.4s, v28.4s, v30.4s\n"  /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W16_V8_RELU                   \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v9.4s, #0\n"               /* for relu */   \
  "fmax   v21.4s, v21.4s, v9.4s\n"   /* relu */       \
  "fmax   v22.4s, v22.4s, v9.4s\n"   /* relu */       \
  "fmax   v23.4s, v23.4s, v9.4s\n"   /* relu */       \
  "fmax   v24.4s, v24.4s, v9.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W8_V8_RELU                    \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v9.4s, #0\n"               /* for relu */   \
  "fmax   v21.4s, v21.4s, v9.4s\n"   /* relu */       \
  "fmax   v22.4s, v22.4s, v9.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W4_V8_RELU                    \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v9.4s, #0\n"               /* for relu */   \
  "fmax   v21.4s, v21.4s, v9.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W48_V8_RELU6                    \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v20.4s, v20.4s, v0.4s\n"    /* relu */        \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmax   v23.4s, v23.4s, v0.4s\n"    /* relu */        \
  "fmax   v24.4s, v24.4s, v0.4s\n"    /* relu */        \
  "fmax   v25.4s, v25.4s, v0.4s\n"    /* relu */        \
  "fmax   v26.4s, v26.4s, v0.4s\n"    /* relu */        \
  "fmax   v27.4s, v27.4s, v0.4s\n"    /* relu */        \
  "fmax   v28.4s, v28.4s, v0.4s\n"    /* relu */        \
  "fmax   v29.4s, v29.4s, v0.4s\n"    /* relu */        \
  "fmax   v30.4s, v30.4s, v0.4s\n"    /* relu */        \
  "fmax   v31.4s, v31.4s, v0.4s\n"    /* relu */        \
  "fmin   v20.4s, v20.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v23.4s, v23.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v24.4s, v24.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v25.4s, v25.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v26.4s, v26.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v27.4s, v27.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v28.4s, v28.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v29.4s, v29.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v30.4s, v30.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v31.4s, v31.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_F32_W32_V8_RELU6                    \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmax   v23.4s, v23.4s, v0.4s\n"    /* relu */        \
  "fmax   v24.4s, v24.4s, v0.4s\n"    /* relu */        \
  "fmax   v25.4s, v25.4s, v0.4s\n"    /* relu */        \
  "fmax   v26.4s, v26.4s, v0.4s\n"    /* relu */        \
  "fmax   v27.4s, v27.4s, v0.4s\n"    /* relu */        \
  "fmax   v28.4s, v28.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v23.4s, v23.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v24.4s, v24.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v25.4s, v25.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v26.4s, v26.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v27.4s, v27.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v28.4s, v28.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_F32_W16_V8_RELU6                    \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmax   v23.4s, v23.4s, v0.4s\n"    /* relu */        \
  "fmax   v24.4s, v24.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v23.4s, v23.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v24.4s, v24.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_F32_W8_V8_RELU6                     \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_F32_W4_V8_RELU6                     \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_F32_W48_V8_LEAKY_RELU                           \
  /* do relu */                                                     \
  "11: \n"                                                          \
  "movi   v0.4s, #0\n"                      /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"             /* leakey relu alpha */ \
  "fcmge  v2.4s,    v20.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v3.4s,    v20.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v4.4s,    v21.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v5.4s,    v21.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v6.4s,    v22.4s,   v0.4s   \n"   /* vcgeq_f32 */         \
  "fmul   v7.4s,    v22.4s,   v1.4s   \n"   /* vmulq_f32 */         \
  "fcmge  v8.4s,    v23.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v9.4s,    v23.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v10.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v11.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v12.4s,    v25.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v13.4s,    v25.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v21.16b,   v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v24.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v25.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "fcmge  v2.4s,    v26.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v3.4s,    v26.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v4.4s,    v27.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v5.4s,    v27.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v6.4s,    v28.4s,   v0.4s   \n"   /* vcgeq_f32 */         \
  "fmul   v7.4s,    v28.4s,   v1.4s   \n"   /* vmulq_f32 */         \
  "fcmge  v8.4s,    v29.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v9.4s,    v29.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v10.4s,    v30.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v11.4s,    v30.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v12.4s,    v31.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v13.4s,    v31.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v26.16b,   v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v27.16b,   v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v28.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v29.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v30.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v31.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "9:\n"

#define SPARSE_INT8_F32_W32_V8_LEAKY_RELU                          \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v23.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v23.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v23.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v24.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "fcmge  v2.4s,    v25.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v25.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v26.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v26.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v27.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v27.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v28.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v28.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v25.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v26.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v27.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v28.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "9:\n"

#define SPARSE_INT8_F32_W16_V8_LEAKY_RELU                          \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v23.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v23.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v23.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v24.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "9:\n"

#define SPARSE_INT8_F32_W8_V8_LEAKY_RELU                           \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "9:\n"

#define SPARSE_INT8_F32_W4_V8_LEAKY_RELU                           \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_INT8_F32_W48_V8_OUT      \
  SPARSE_INT8_F32_W48_V8_KERNEL         \
  SPARSE_INT8_F32_W48_V8_RELU           \
  SPARSE_INT8_F32_W48_V8_RELU6          \
  SPARSE_INT8_F32_W48_V8_LEAKY_RELU     \
  /* store result */                    \
  "stp   q20, q21,  [%[c_ptr]]\n"       \
  "stp   q22, q23,  [%[c_ptr], #32]\n"  \
  "stp   q24, q25,  [%[c_ptr], #64]\n"  \
  "stp   q26, q27,  [%[c_ptr], #96]\n"  \
  "stp   q28, q29,  [%[c_ptr], #128]\n" \
  "stp   q30, q31,  [%[c_ptr], #160]\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_INT8_F32_W32_V8_OUT     \
  SPARSE_INT8_F32_W32_V8_KERNEL        \
  SPARSE_INT8_F32_W32_V8_RELU          \
  SPARSE_INT8_F32_W32_V8_RELU6         \
  SPARSE_INT8_F32_W32_V8_LEAKY_RELU    \
  /* store result */                   \
  "stp   q21, q22,  [%[c_ptr]]\n"      \
  "stp   q23, q24,  [%[c_ptr], #32]\n" \
  "stp   q25, q26,  [%[c_ptr], #64]\n" \
  "stp   q27, q28,  [%[c_ptr], #96]\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_INT8_F32_W16_V8_OUT  \
  SPARSE_INT8_F32_W16_V8_KERNEL     \
  SPARSE_INT8_F32_W16_V8_RELU       \
  SPARSE_INT8_F32_W16_V8_RELU6      \
  SPARSE_INT8_F32_W16_V8_LEAKY_RELU \
  /* store result */                \
  "stp   q21, q22,  [%[c_ptr]]\n"   \
  "stp   q23, q24,  [%[c_ptr], #32]\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_INT8_F32_W8_V8_OUT  \
  SPARSE_INT8_F32_W8_V8_KERNEL     \
  SPARSE_INT8_F32_W8_V8_RELU       \
  SPARSE_INT8_F32_W8_V8_RELU6      \
  SPARSE_INT8_F32_W8_V8_LEAKY_RELU \
  /* store result */               \
  "stp   q21, q22,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_INT8_F32_W4_V8_OUT  \
  SPARSE_INT8_F32_W4_V8_KERNEL     \
  SPARSE_INT8_F32_W4_V8_RELU       \
  SPARSE_INT8_F32_W4_V8_RELU6      \
  SPARSE_INT8_F32_W4_V8_LEAKY_RELU \
  /* store result */               \
  "str   q21,  [%[c_ptr]]\n"

/**
 * \brief Sparse calculation implementation of 1x1 convolution, the input-output
 * type is int8-f32.
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
void sparse_conv_int8_fp32_pipelined(const int8_t* A,
                                     const int8_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float* bias,
                                     const float* scale,
                                     float* output,
                                     int M,
                                     int K,
                                     int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  int flag_bias = (bias != nullptr) ? 1 : 0;
  size_t mc = N * sizeof(int8_t);
  size_t nc = M;
  size_t output_stride = N * sizeof(float);
  size_t output_decrement = output_stride * nc - 48 * sizeof(float);

  while
    SPARSE_LIKELY(mc >= 48 * sizeof(int8_t)) {
      const int8_t* w = A;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      const float* sc = scale;

      for (size_t i = 0; i < nc; i++) {
        uint32_t nnz = *nnzmap++;
        float vsclae = *sc++;
        float valpha = alpha;
        float vbias = (bias != nullptr) ? bias[i] : 0.0;
        // clang-format off
          asm volatile(SPARSE_INT8_F32_W48_V8_OUT
            : [a_ptr] "+r"(w),
              [b_ptr] "+r"(B),
              [c_ptr] "+r"(output),
              [k] "+r"(nnz),
              [widx_dmap] "+r"(dmap)
            : [vscale] "r"(vsclae),
              [vbias] "r"(vbias),
              [vflag_act] "r"(flag_act),
              [valpha] "r"(valpha)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w1", "x1", "cc", "memory");
        // clang-format on
        output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
      }
      output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
      B += 48;
      mc -= 48 * sizeof(int8_t);
    }
  if
    SPARSE_UNLIKELY(mc != 0) {
      output_decrement += 16 * sizeof(float);
      if (mc & (32 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_F32_W32_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 32;
        mc -= 32 * sizeof(int8_t);
      }
      output_decrement += 16 * sizeof(float);
      if (mc & (16 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_F32_W16_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v9", "v11", "v12", "v13", "v14", "v21", "v22", "v23",
                  "v24", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 16;
        mc -= 16 * sizeof(int8_t);
      }
      output_decrement += 8 * sizeof(float);
      if (mc & (8 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_F32_W8_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v9", "v11", "v12", "v21", 
              "v22", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 8;
        mc -= 8 * sizeof(int8_t);
      }
      output_decrement += 4 * sizeof(float);
      if (mc & (4 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_F32_W4_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "v0", "v1", "v2", "v3", "v4", "v9", "v11", "v21", 
              "w1", "w2", "w3", "w4", "w5", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 4;
        mc -= 4 * sizeof(int8_t);
      }

      if
        SPARSE_UNLIKELY(mc != 0 && mc < 4 * sizeof(int8_t)) {
          const int8_t* w = A;
          const int32_t* dmap = widx_dmap;
          const uint32_t* nnzmap = nidx_nnzmap;
          const float* bs = bias;
          const float* sc = scale;
          // const float* al = alpha;
          float val = alpha;
          int mindex = mc / sizeof(int8_t);

          for (size_t i = 0; i < nc; i++) {
            float vbias = (bias != nullptr) ? *bs++ : 0;
            float vscale = *sc++;
            for (size_t k = 0; k < mc; k++) {
              *(output + k) = 0;
            }
            uint32_t nnz = *nnzmap++;
            for (size_t j = 0; j < nnz; j++) {
              for (size_t k = 0; k < mc; k++) {
                *(output + k) += (*w) * (*(B + k));
              }
              w += 1;
              intptr_t diff = *dmap++;
              B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
            }
            switch (flag_act) {
              case 0:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) = *(output + k) * vscale + vbias;
                }
                break;
              case 1:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) = *(output + k) * vscale + vbias;
                  *(output + k) = *(output + k) > 0 ? *(output + k) : 0;
                }
                break;
              case 2:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) = *(output + k) * vscale + vbias;
                  *(output + k) = *(output + k) > 0 ? *(output + k) : 0;
                  *(output + k) = *(output + k) < val ? *(output + k) : val;
                }
                break;
              default:
                for (size_t k = 0; k < mindex; k++) {
                  *(output + k) = *(output + k) * vscale + vbias;
                  *(output + k) =
                      *(output + k) >= 0 ? *(output + k) : *(output + k) * val;
                }
                break;
            }
            output =
                reinterpret_cast<float*>((uintptr_t)output + output_stride);
          }
        }
    }
}

#define SPARSE_INT8_INT8_W48_V8_KERNEL           \
  "eor v8.16b, v0.16b, v0.16b\n"                 \
  "eor v9.16b, v1.16b, v1.16b\n"                 \
  "eor v10.16b, v2.16b, v2.16b\n"                \
  "eor v11.16b, v3.16b, v3.16b\n"                \
  "eor v12.16b, v4.16b, v4.16b\n"                \
  "prfm  pldl1keep, [%[a_ptr], #32]\n"           \
  "eor v13.16b, v5.16b, v5.16b\n"                \
  "eor v14.16b, v6.16b, v6.16b\n"                \
  "prfm  pldl1keep, [%[widx_dmap], #32]\n"       \
  "eor v15.16b, v7.16b, v7.16b\n"                \
  "eor v16.16b, v0.16b, v0.16b\n"                \
  "prfm  pldl1keep, [%[b_ptr], #48]\n"           \
  "eor v17.16b, v1.16b, v1.16b\n"                \
  "eor v18.16b, v2.16b, v2.16b\n"                \
  "eor v19.16b, v3.16b, v3.16b\n"                \
  "dup     v20.4s,  %w[vbias]\n"                 \
  "dup     v21.4s,  v20.s[0]\n"                  \
  "dup     v22.4s,  v20.s[0]\n"                  \
  "dup     v23.4s,  v20.s[0]\n"                  \
  "dup     v24.4s,  v20.s[0]\n"                  \
  "dup     v25.4s,  v20.s[0]\n"                  \
  "dup     v26.4s,  v20.s[0]\n"                  \
  "dup     v27.4s,  v20.s[0]\n"                  \
  "dup     v28.4s,  v20.s[0]\n"                  \
  "dup     v29.4s,  v20.s[0]\n"                  \
  "dup     v30.4s,  v20.s[0]\n"                  \
  "dup     v31.4s,  v20.s[0]\n"                  \
  "cbz    %w[k],    1f\n"                        \
  "0:\n"                                         \
  "ld1r  {v0.16b}, [%[a_ptr]], #1\n"             \
  "ldr   w1, [%[widx_dmap]],   #4\n"             \
  "sxtw  x1,  w1\n"                              \
  "ld1   {v1.16b, v2.16b, v3.16b}, [%[b_ptr]]\n" \
  "add   %[b_ptr],  %[b_ptr], x1\n"              \
  "smull   v4.8h,   v0.8b,   v1.8b\n"            \
  "smull2  v5.8h,   v0.16b,  v1.16b\n"           \
  "smull   v6.8h,   v0.8b,   v2.8b\n"            \
  "smull2  v7.8h,   v0.16b,  v2.16b\n"           \
  "subs    %w[k],   %w[k],   #1\n"               \
  "saddw   v8.4s,  v8.4s,  v4.4h\n"              \
  "saddw2  v9.4s,  v9.4s,  v4.8h\n"              \
  "prfm  pldl1keep, [%[b_ptr], #48]\n"           \
  "saddw   v10.4s,  v10.4s,  v5.4h\n"            \
  "saddw2  v11.4s,  v11.4s,  v5.8h\n"            \
  "saddw   v12.4s,  v12.4s,  v6.4h\n"            \
  "saddw2  v13.4s,  v13.4s,  v6.8h\n"            \
  "saddw   v14.4s,  v14.4s,  v7.4h\n"            \
  "saddw2  v15.4s,  v15.4s,  v7.8h\n"            \
  "smull   v4.8h,   v0.8b,   v3.8b\n"            \
  "smull2  v5.8h,   v0.16b,  v3.16b\n"           \
  "saddw   v16.4s,  v16.4s,  v4.4h\n"            \
  "saddw2  v17.4s,  v17.4s,  v4.8h\n"            \
  "saddw   v18.4s,  v18.4s,  v5.4h\n"            \
  "saddw2  v19.4s,  v19.4s,  v5.8h\n"            \
  "bne     0b\n"                                 \
  "1:\n"                                         \
  "dup     v0.4s,  %w[vscale]\n"                 \
  "scvtf   v1.4s,  v8.4s\n"                      \
  "scvtf   v2.4s,  v9.4s\n"                      \
  "scvtf   v3.4s,  v10.4s\n"                     \
  "scvtf   v4.4s,  v11.4s\n"                     \
  "scvtf   v5.4s,  v12.4s\n"                     \
  "scvtf   v6.4s,  v13.4s\n"                     \
  "scvtf   v7.4s,  v14.4s\n" /* scale */         \
  "fmla    v20.4s,  v1.4s,  v0.s[0]\n"           \
  "fmla    v21.4s,  v2.4s,  v0.s[0]\n"           \
  "fmla    v22.4s,  v3.4s,  v0.s[0]\n"           \
  "fmla    v23.4s,  v4.4s,  v0.s[0]\n"           \
  "fmla    v24.4s,  v5.4s,  v0.s[0]\n"           \
  "fmla    v25.4s,  v6.4s,  v0.s[0]\n"           \
  "fmla    v26.4s,  v7.4s,  v0.s[0]\n"           \
  "scvtf   v1.4s,  v15.4s\n"                     \
  "scvtf   v2.4s,  v16.4s\n"                     \
  "scvtf   v3.4s,  v17.4s\n"                     \
  "scvtf   v4.4s,  v18.4s\n"                     \
  "scvtf   v5.4s,  v19.4s\n" /* scale */         \
  "fmla    v27.4s,  v1.4s,  v0.s[0]\n"           \
  "fmla    v28.4s,  v2.4s,  v0.s[0]\n"           \
  "fmla    v29.4s,  v3.4s,  v0.s[0]\n"           \
  "fmla    v30.4s,  v4.4s,  v0.s[0]\n"           \
  "fmla    v31.4s,  v5.4s,  v0.s[0]\n"

#define SPARSE_INT8_INT8_W32_V8_KERNEL     \
  "eor v11.16b, v0.16b, v0.16b\n"          \
  "eor v12.16b, v1.16b, v1.16b\n"          \
  "prfm  pldl1keep, [%[a_ptr], #32]\n"     \
  "eor v13.16b, v2.16b, v2.16b\n"          \
  "eor v14.16b, v3.16b, v3.16b\n"          \
  "prfm  pldl1keep, [%[widx_dmap], #32]\n" \
  "eor v15.16b, v4.16b, v4.16b\n"          \
  "eor v16.16b, v5.16b, v5.16b\n"          \
  "prfm  pldl1keep, [%[b_ptr], #32]\n"     \
  "eor v17.16b, v6.16b, v6.16b\n"          \
  "eor v18.16b, v7.16b, v7.16b\n"          \
  "dup     v21.4s,  %w[vbias]\n"           \
  "dup     v22.4s,  v21.s[0]\n"            \
  "dup     v23.4s,  v21.s[0]\n"            \
  "dup     v24.4s,  v21.s[0]\n"            \
  "dup     v25.4s,  v21.s[0]\n"            \
  "dup     v26.4s,  v21.s[0]\n"            \
  "dup     v27.4s,  v21.s[0]\n"            \
  "dup     v28.4s,  v21.s[0]\n"            \
  "cbz    %w[k],    1f\n"                  \
  "0:\n"                                   \
  "ld1r  {v0.16b}, [%[a_ptr]], #1\n"       \
  "ldr   w1, [%[widx_dmap]],   #4\n"       \
  "sxtw  x1,  w1\n"                        \
  "ld1   {v1.16b, v2.16b}, [%[b_ptr]]\n"   \
  "add   %[b_ptr],  %[b_ptr], x1\n"        \
  "smull   v3.8h,   v0.8b,   v1.8b\n"      \
  "smull   v5.8h,   v0.8b,   v2.8b\n"      \
  "subs    %w[k],   %w[k],   #1\n"         \
  "smull2  v4.8h,   v0.16b,  v1.16b\n"     \
  "smull2  v6.8h,   v0.16b,  v2.16b\n"     \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"      \
  "saddw   v13.4s,  v13.4s,  v4.4h\n"      \
  "prfm  pldl1keep, [%[b_ptr], #32]\n"     \
  "saddw   v15.4s,  v15.4s,  v5.4h\n"      \
  "saddw   v17.4s,  v17.4s,  v6.4h\n"      \
  "saddw2  v12.4s,  v12.4s,  v3.8h\n"      \
  "saddw2  v14.4s,  v14.4s,  v4.8h\n"      \
  "saddw2  v16.4s,  v16.4s,  v5.8h\n"      \
  "saddw2  v18.4s,  v18.4s,  v6.8h\n"      \
  "bne     0b\n"                           \
  "1:\n"                                   \
  "scvtf   v3.4s,  v11.4s\n"               \
  "scvtf   v4.4s,  v12.4s\n"               \
  "scvtf   v5.4s,  v13.4s\n"               \
  "scvtf   v6.4s,  v14.4s\n"               \
  "scvtf   v7.4s,  v15.4s\n"               \
  "scvtf   v8.4s,  v16.4s\n"               \
  "scvtf   v9.4s,  v17.4s\n"               \
  "scvtf   v10.4s, v18.4s\n" /* scale */   \
  "dup     v31.4s,  %w[vscale]\n"          \
  "fmla    v21.4s,  v3.4s,  v31.s[0]\n"    \
  "fmla    v22.4s,  v4.4s,  v31.s[0]\n"    \
  "fmla    v23.4s,  v5.4s,  v31.s[0]\n"    \
  "fmla    v24.4s,  v6.4s,  v31.s[0]\n"    \
  "fmla    v25.4s,  v7.4s,  v31.s[0]\n"    \
  "fmla    v26.4s,  v8.4s,  v31.s[0]\n"    \
  "fmla    v27.4s,  v9.4s,  v31.s[0]\n"    \
  "fmla    v28.4s,  v10.4s, v31.s[0]\n"

#define SPARSE_INT8_INT8_W16_V8_KERNEL   \
  "eor v11.16b, v0.16b, v0.16b\n"        \
  "eor v12.16b, v1.16b, v1.16b\n"        \
  "eor v13.16b, v2.16b, v2.16b\n"        \
  "prfm  pldl1keep, [%[b_ptr], #16]\n"   \
  "eor v14.16b, v3.16b, v3.16b\n"        \
  "dup     v21.4s,  %w[vbias]\n"         \
  "dup     v22.4s,  v21.s[0]\n"          \
  "dup     v23.4s,  v21.s[0]\n"          \
  "dup     v24.4s,  v21.s[0]\n"          \
  "cbz    %w[k],    1f\n"                \
  "0:\n"                                 \
  "ld1r  {v0.16b}, [%[a_ptr]], #1\n"     \
  "ldr   w1, [%[widx_dmap]],   #4\n"     \
  "sxtw  x1,  w1\n"                      \
  "ld1   {v1.16b}, [%[b_ptr]]\n"         \
  "add   %[b_ptr],  %[b_ptr], x1\n"      \
  "smull   v3.8h,   v0.8b,   v1.8b\n"    \
  "smull2  v4.8h,   v0.16b,  v1.16b\n"   \
  "prfm  pldl1keep, [%[b_ptr], #16]\n"   \
  "subs    %w[k],   %w[k],   #1\n"       \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"    \
  "saddw2  v12.4s,  v12.4s,  v3.8h\n"    \
  "saddw   v13.4s,  v13.4s,  v4.4h\n"    \
  "saddw2  v14.4s,  v14.4s,  v4.8h\n"    \
  "bne     0b\n"                         \
  "1:\n"                                 \
  "scvtf   v5.4s,  v11.4s\n"             \
  "scvtf   v6.4s,  v12.4s\n"             \
  "scvtf   v7.4s,  v13.4s\n"             \
  "scvtf   v8.4s,  v14.4s\n" /* scale */ \
  "dup     v2.4s,  %w[vscale]\n"         \
  "fmla    v21.4s,  v5.4s,  v2.s[0]\n"   \
  "fmla    v22.4s,  v6.4s,  v2.s[0]\n"   \
  "fmla    v23.4s,  v7.4s,  v2.s[0]\n"   \
  "fmla    v24.4s,  v8.4s,  v2.s[0]\n"

#define SPARSE_INT8_INT8_W8_V8_KERNEL    \
  "eor v11.16b, v0.16b, v0.16b\n"        \
  "eor v12.16b, v1.16b, v1.16b\n"        \
  "dup     v21.4s,  %w[vbias]\n"         \
  "dup     v22.4s,  v21.s[0]\n"          \
  "cbz    %w[k],    1f\n"                \
  "0:\n"                                 \
  "ld1r  {v0.8b}, [%[a_ptr]], #1\n"      \
  "ldr   w1, [%[widx_dmap]],   #4\n"     \
  "sxtw  x1,  w1\n"                      \
  "ld1   {v1.8b}, [%[b_ptr]]\n"          \
  "add   %[b_ptr],  %[b_ptr], x1\n"      \
  "smull   v3.8h,   v0.8b,   v1.8b\n"    \
  "subs    %w[k],   %w[k],   #1\n"       \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"    \
  "saddw2  v12.4s,  v12.4s,  v3.8h\n"    \
  "bne     0b\n"                         \
  "1:\n"                                 \
  "scvtf   v4.4s,  v11.4s\n"             \
  "scvtf   v5.4s,  v12.4s\n" /* scale */ \
  "dup     v2.4s,   %w[vscale]\n"        \
  "fmla    v21.4s,  v4.4s,  v2.s[0]\n"   \
  "fmla    v22.4s,  v5.4s,  v2.s[0]\n"

#define SPARSE_INT8_INT8_W4_V8_KERNEL    \
  "eor v11.16b, v0.16b, v0.16b\n"        \
  "dup     v21.4s,  %w[vbias]\n"         \
  "cbz    %w[k],    1f\n"                \
  "0:\n"                                 \
  "ld1r  {v0.8b}, [%[a_ptr]], #1\n"      \
  "ldr   w1, [%[widx_dmap]],   #4\n"     \
  "ldrsb   w2, [%[b_ptr]]\n"             \
  "ldrsb   w3, [%[b_ptr], #1]\n"         \
  "ldrsb   w4, [%[b_ptr], #2]\n"         \
  "ldrsb   w5, [%[b_ptr], #3]\n"         \
  "sxtw  x1,  w1\n"                      \
  "mov   v1.b[0], w2\n"                  \
  "mov   v1.b[1], w3\n"                  \
  "mov   v1.b[2], w4\n"                  \
  "mov   v1.b[3], w5\n"                  \
  "add   %[b_ptr],  %[b_ptr], x1\n"      \
  "smull   v3.8h,   v0.8b,   v1.8b\n"    \
  "subs    %w[k],   %w[k],   #1\n"       \
  "saddw   v11.4s,  v11.4s,  v3.4h\n"    \
  "bne     0b\n"                         \
  "1:\n"                                 \
  "scvtf   v4.4s,  v11.4s\n" /* scale */ \
  "dup     v2.4s,   %w[vscale]\n"        \
  "fmla    v21.4s,  v4.4s,  v2.s[0]\n"

#define SPARSE_INT8_INT8_W48_V8_RELU                  \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v0.4s, #0\n"               /* for relu */   \
  "fmax   v20.4s, v20.4s, v0.4s\n"   /* relu */       \
  "fmax   v21.4s, v21.4s, v0.4s\n"   /* relu */       \
  "fmax   v22.4s, v22.4s, v0.4s\n"   /* relu */       \
  "fmax   v23.4s, v23.4s, v0.4s\n"   /* relu */       \
  "fmax   v24.4s, v24.4s, v0.4s\n"   /* relu */       \
  "fmax   v25.4s, v25.4s, v0.4s\n"   /* relu */       \
  "fmax   v26.4s, v26.4s, v0.4s\n"   /* relu */       \
  "fmax   v27.4s, v27.4s, v0.4s\n"   /* relu */       \
  "fmax   v28.4s, v28.4s, v0.4s\n"   /* relu */       \
  "fmax   v29.4s, v29.4s, v0.4s\n"   /* relu */       \
  "fmax   v30.4s, v30.4s, v0.4s\n"   /* relu */       \
  "fmax   v31.4s, v31.4s, v0.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W32_V8_RELU                  \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v30.4s, #0\n"              /* for relu */   \
  "fmax   v21.4s, v21.4s, v30.4s\n"  /* relu */       \
  "fmax   v22.4s, v22.4s, v30.4s\n"  /* relu */       \
  "fmax   v23.4s, v23.4s, v30.4s\n"  /* relu */       \
  "fmax   v24.4s, v24.4s, v30.4s\n"  /* relu */       \
  "fmax   v25.4s, v25.4s, v30.4s\n"  /* relu */       \
  "fmax   v26.4s, v26.4s, v30.4s\n"  /* relu */       \
  "fmax   v27.4s, v27.4s, v30.4s\n"  /* relu */       \
  "fmax   v28.4s, v28.4s, v30.4s\n"  /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W16_V8_RELU                  \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v9.4s, #0\n"               /* for relu */   \
  "fmax   v21.4s, v21.4s, v9.4s\n"   /* relu */       \
  "fmax   v22.4s, v22.4s, v9.4s\n"   /* relu */       \
  "fmax   v23.4s, v23.4s, v9.4s\n"   /* relu */       \
  "fmax   v24.4s, v24.4s, v9.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W8_V8_RELU                   \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v9.4s, #0\n"               /* for relu */   \
  "fmax   v21.4s, v21.4s, v9.4s\n"   /* relu */       \
  "fmax   v22.4s, v22.4s, v9.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W4_V8_RELU                   \
  /* do relu */                                       \
  "cmp    %w[vflag_act],    #0\n"    /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %w[vflag_act],    #1\n"    /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "movi   v9.4s, #0\n"               /* for relu */   \
  "fmax   v21.4s, v21.4s, v9.4s\n"   /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W48_V8_RELU6                   \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v20.4s, v20.4s, v0.4s\n"    /* relu */        \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmax   v23.4s, v23.4s, v0.4s\n"    /* relu */        \
  "fmax   v24.4s, v24.4s, v0.4s\n"    /* relu */        \
  "fmax   v25.4s, v25.4s, v0.4s\n"    /* relu */        \
  "fmax   v26.4s, v26.4s, v0.4s\n"    /* relu */        \
  "fmax   v27.4s, v27.4s, v0.4s\n"    /* relu */        \
  "fmax   v28.4s, v28.4s, v0.4s\n"    /* relu */        \
  "fmax   v29.4s, v29.4s, v0.4s\n"    /* relu */        \
  "fmax   v30.4s, v30.4s, v0.4s\n"    /* relu */        \
  "fmax   v31.4s, v31.4s, v0.4s\n"    /* relu */        \
  "fmin   v20.4s, v20.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v23.4s, v23.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v24.4s, v24.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v25.4s, v25.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v26.4s, v26.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v27.4s, v27.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v28.4s, v28.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v29.4s, v29.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v30.4s, v30.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v31.4s, v31.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_INT8_W32_V8_RELU6                   \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmax   v23.4s, v23.4s, v0.4s\n"    /* relu */        \
  "fmax   v24.4s, v24.4s, v0.4s\n"    /* relu */        \
  "fmax   v25.4s, v25.4s, v0.4s\n"    /* relu */        \
  "fmax   v26.4s, v26.4s, v0.4s\n"    /* relu */        \
  "fmax   v27.4s, v27.4s, v0.4s\n"    /* relu */        \
  "fmax   v28.4s, v28.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v23.4s, v23.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v24.4s, v24.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v25.4s, v25.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v26.4s, v26.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v27.4s, v27.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v28.4s, v28.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_INT8_W16_V8_RELU6                   \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmax   v23.4s, v23.4s, v0.4s\n"    /* relu */        \
  "fmax   v24.4s, v24.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v23.4s, v23.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v24.4s, v24.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_INT8_W8_V8_RELU6                    \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmax   v22.4s, v22.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "fmin   v22.4s, v22.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_INT8_W4_V8_RELU6                    \
  /* do relu6 */                                        \
  "10: \n"                                              \
  "cmp   %w[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n"  /* no act end */  \
  "movi   v0.4s, #0\n"                /* for relu6 */   \
  "dup    v1.4s,  %w[valpha]\n"       /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n"    /* relu */        \
  "fmin   v21.4s, v21.4s, v1.4s\n"    /* relu6 */       \
  "b      9f                    \n"   /* relu end */

#define SPARSE_INT8_INT8_W48_V8_LEAKY_RELU                          \
  /* do relu */                                                     \
  "11: \n"                                                          \
  "movi   v0.4s, #0\n"                      /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"             /* leakey relu alpha */ \
  "fcmge  v2.4s,    v20.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v3.4s,    v20.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v4.4s,    v21.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v5.4s,    v21.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v6.4s,    v22.4s,   v0.4s   \n"   /* vcgeq_f32 */         \
  "fmul   v7.4s,    v22.4s,   v1.4s   \n"   /* vmulq_f32 */         \
  "fcmge  v8.4s,    v23.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v9.4s,    v23.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v10.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v11.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v12.4s,    v25.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v13.4s,    v25.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v20.16b,   v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v21.16b,   v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v22.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v23.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v24.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v25.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "fcmge  v2.4s,    v26.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v3.4s,    v26.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v4.4s,    v27.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v5.4s,    v27.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v6.4s,    v28.4s,   v0.4s   \n"   /* vcgeq_f32 */         \
  "fmul   v7.4s,    v28.4s,   v1.4s   \n"   /* vmulq_f32 */         \
  "fcmge  v8.4s,    v29.4s,    v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v9.4s,    v29.4s,    v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v10.4s,    v30.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v11.4s,    v30.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v12.4s,    v31.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v13.4s,    v31.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v26.16b,   v3.16b,   v2.16b  \n"  /* choose*/             \
  "bif    v27.16b,   v5.16b,   v4.16b  \n"  /* choose*/             \
  "bif    v28.16b,  v7.16b,   v6.16b  \n"   /* choose*/             \
  "bif    v29.16b,  v9.16b,   v8.16b  \n"   /* choose*/             \
  "bif    v30.16b,  v11.16b,   v10.16b  \n" /* choose*/             \
  "bif    v31.16b,  v13.16b,   v12.16b  \n" /* choose*/             \
  "9:\n"

#define SPARSE_INT8_INT8_W32_V8_LEAKY_RELU                         \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v23.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v23.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v23.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v24.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "fcmge  v2.4s,    v25.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v25.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v26.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v26.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v27.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v27.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v28.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v28.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v25.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v26.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v27.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v28.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "9:\n"

#define SPARSE_INT8_INT8_W16_V8_LEAKY_RELU                         \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,    v23.4s,   v0.4s   \n"  /* vcgeq_f32 */         \
  "fmul   v7.4s,    v23.4s,   v1.4s   \n"  /* vmulq_f32 */         \
  "fcmge  v8.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "bif    v23.16b,  v7.16b,   v6.16b  \n"  /* choose*/             \
  "bif    v24.16b,  v9.16b,   v8.16b  \n"  /* choose*/             \
  "9:\n"

#define SPARSE_INT8_INT8_W8_V8_LEAKY_RELU                          \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "fcmge  v4.4s,    v22.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,    v22.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/             \
  "9:\n"

#define SPARSE_INT8_INT8_W4_V8_LEAKY_RELU                          \
  /* do relu */                                                    \
  "11: \n"                                                         \
  "movi   v0.4s, #0\n"                     /* for relu6 */         \
  "dup    v1.4s,  %w[valpha]\n"            /* leakey relu alpha */ \
  "fcmge  v2.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */         \
  "fmul   v3.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */         \
  "bif    v21.16b,   v3.16b,   v2.16b  \n" /* choose*/             \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_INT8_INT8_W48_V8_OUT                                      \
  SPARSE_INT8_INT8_W48_V8_KERNEL                                         \
  SPARSE_INT8_INT8_W48_V8_RELU                                           \
  SPARSE_INT8_INT8_W48_V8_RELU6                                          \
  SPARSE_INT8_INT8_W48_V8_LEAKY_RELU                                     \
  /* store result */                                                     \
  "ld1    {v12.4s},   [%[vmax]] \n" /* v8 = -127 */ /* data >= -127 */   \
  "fcmge v0.4s,  v20.4s, v12.4s\n"                                       \
  "fcmge v1.4s,  v21.4s, v12.4s\n"                                       \
  "fcmge v2.4s,  v22.4s, v12.4s\n"                                       \
  "fcmge v3.4s,  v23.4s, v12.4s\n"                                       \
  "fcmge v4.4s,  v24.4s, v12.4s\n"                                       \
  "fcmge v5.4s,  v25.4s, v12.4s\n"                                       \
  "fcmge v6.4s,  v26.4s, v12.4s\n"                                       \
  "fcmge v7.4s,  v27.4s, v12.4s\n"                                       \
  "fcmge v8.4s,  v28.4s, v12.4s\n"                                       \
  "fcmge v9.4s,  v29.4s, v12.4s\n"                                       \
  "fcmge v10.4s, v30.4s, v12.4s\n"                                       \
  "fcmge v11.4s, v31.4s, v12.4s\n" /* choose data */                     \
  "bif v20.16b, v12.16b, v0.16b           \n"                            \
  "bif v21.16b, v12.16b, v1.16b           \n"                            \
  "bif v22.16b, v12.16b, v2.16b           \n"                            \
  "bif v23.16b, v12.16b, v3.16b           \n"                            \
  "bif v24.16b, v12.16b, v4.16b           \n"                            \
  "bif v25.16b, v12.16b, v5.16b           \n"                            \
  "bif v26.16b, v12.16b, v6.16b           \n"                            \
  "bif v27.16b, v12.16b, v7.16b           \n"                            \
  "bif v28.16b, v12.16b, v8.16b           \n"                            \
  "bif v29.16b, v12.16b, v9.16b           \n"                            \
  "bif v30.16b, v12.16b, v10.16b          \n"                            \
  "bif v31.16b, v12.16b, v11.16b          \n"                            \
  "fcvtas v0.4s, v20.4s\n"   /*  cvt to int */                           \
  "fcvtas v1.4s, v21.4s\n"   /*  cvt to int */                           \
  "fcvtas v2.4s, v22.4s\n"   /*  cvt to int */                           \
  "fcvtas v3.4s, v23.4s\n"   /*  cvt to int */                           \
  "fcvtas v4.4s, v24.4s\n"   /*  cvt to int */                           \
  "fcvtas v5.4s, v25.4s\n"   /*  cvt to int */                           \
  "fcvtas v6.4s, v26.4s\n"   /*  cvt to int */                           \
  "fcvtas v7.4s, v27.4s\n"   /*  cvt to int */                           \
  "fcvtas v8.4s, v28.4s\n"   /*  cvt to int */                           \
  "fcvtas v9.4s, v29.4s\n"   /*  cvt to int */                           \
  "fcvtas v10.4s, v30.4s\n"  /*  cvt to int */                           \
  "fcvtas v11.4s, v31.4s\n"  /*  cvt to int */                           \
  "sqxtn  v14.4h, v0.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v14.8h, v1.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v15.4h, v2.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v15.8h, v3.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v16.4h, v4.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v16.8h, v5.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v17.4h, v6.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v17.8h, v7.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v18.4h, v8.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v18.8h, v9.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v19.4h, v10.4s\n"  /*  cvt int32 to int16 */                   \
  "sqxtn2 v19.8h, v11.4s\n"  /*  cvt int32 to int16 */                   \
  "sqxtn  v21.8b, v14.8h\n"  /*  cvt int16 to int8 */                    \
  "sqxtn2 v21.16b, v15.8h\n" /*  cvt int16 to int8 */                    \
  "sqxtn  v22.8b, v16.8h\n"  /*  cvt int16 to int8 */                    \
  "sqxtn2 v22.16b, v17.8h\n" /*  cvt int16 to int8 */                    \
  "sqxtn  v23.8b, v18.8h\n"  /*  cvt int16 to int8 */                    \
  "sqxtn2 v23.16b, v19.8h\n" /*  cvt int16 to int8 */ /* store result */ \
  "stp   q21, q22,  [%[c_ptr]]\n"                                        \
  "str   q23,  [%[c_ptr], #32]\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_INT8_INT8_W32_V8_OUT                                      \
  SPARSE_INT8_INT8_W32_V8_KERNEL                                         \
  SPARSE_INT8_INT8_W32_V8_RELU                                           \
  SPARSE_INT8_INT8_W32_V8_RELU6                                          \
  SPARSE_INT8_INT8_W32_V8_LEAKY_RELU                                     \
  /* store result */                                                     \
  "ld1    {v8.4s},   [%[vmax]] \n" /* v8 = -127 */ /* data >= -127 */    \
  "fcmge v0.4s, v21.4s, v8.4s\n"                                         \
  "fcmge v1.4s, v22.4s, v8.4s\n"                                         \
  "fcmge v2.4s, v23.4s, v8.4s\n"                                         \
  "fcmge v3.4s, v24.4s, v8.4s\n"                                         \
  "fcmge v4.4s, v25.4s, v8.4s\n"                                         \
  "fcmge v5.4s, v26.4s, v8.4s\n"                                         \
  "fcmge v6.4s, v27.4s, v8.4s\n"                                         \
  "fcmge v7.4s, v28.4s, v8.4s\n" /* choose data */                       \
  "bif v21.16b,  v8.16b, v0.16b           \n"                            \
  "bif v22.16b, v8.16b, v1.16b            \n"                            \
  "bif v23.16b, v8.16b, v2.16b            \n"                            \
  "bif v24.16b, v8.16b, v3.16b            \n"                            \
  "bif v25.16b, v8.16b, v4.16b            \n"                            \
  "bif v26.16b, v8.16b, v5.16b            \n"                            \
  "bif v27.16b, v8.16b, v6.16b            \n"                            \
  "bif v28.16b, v8.16b, v7.16b            \n"                            \
  "fcvtas v0.4s, v21.4s\n"   /*  cvt to int */                           \
  "fcvtas v1.4s, v22.4s\n"   /*  cvt to int */                           \
  "fcvtas v2.4s, v23.4s\n"   /*  cvt to int */                           \
  "fcvtas v3.4s, v24.4s\n"   /*  cvt to int */                           \
  "fcvtas v4.4s, v25.4s\n"   /*  cvt to int */                           \
  "fcvtas v5.4s, v26.4s\n"   /*  cvt to int */                           \
  "fcvtas v6.4s, v27.4s\n"   /*  cvt to int */                           \
  "fcvtas v7.4s, v28.4s\n"   /*  cvt to int */                           \
  "sqxtn  v16.4h, v0.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v16.8h, v1.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v17.4h, v2.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v17.8h, v3.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v18.4h, v4.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v18.8h, v5.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v19.4h, v6.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn2 v19.8h, v7.4s\n"   /*  cvt int32 to int16 */                   \
  "sqxtn  v21.8b, v16.8h\n"  /*  cvt int16 to int8 */                    \
  "sqxtn2 v21.16b, v17.8h\n" /*  cvt int16 to int8 */                    \
  "sqxtn  v22.8b, v18.8h\n"  /*  cvt int16 to int8 */                    \
  "sqxtn2 v22.16b, v19.8h\n" /*  cvt int16 to int8 */ /* store result */ \
  "stp   q21, q22,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_INT8_INT8_W16_V8_OUT                                      \
  SPARSE_INT8_INT8_W16_V8_KERNEL                                         \
  SPARSE_INT8_INT8_W16_V8_RELU                                           \
  SPARSE_INT8_INT8_W16_V8_RELU6                                          \
  SPARSE_INT8_INT8_W16_V8_LEAKY_RELU                                     \
  "ld1    {v8.4s},   [%[vmax]] \n" /* v8 = -127 */ /* data >= -127 */    \
  "fcmge v0.4s, v21.4s, v8.4s\n"                                         \
  "fcmge v1.4s, v22.4s, v8.4s\n"                                         \
  "fcmge v2.4s, v23.4s, v8.4s\n"                                         \
  "fcmge v3.4s, v24.4s, v8.4s\n" /* choose data */                       \
  "bif v21.16b,  v8.16b, v0.16b           \n"                            \
  "bif v22.16b, v8.16b, v1.16b            \n"                            \
  "bif v23.16b, v8.16b, v2.16b            \n"                            \
  "bif v24.16b, v8.16b, v3.16b            \n"                            \
  "fcvtas v0.4s, v21.4s\n"  /*  cvt to int */                            \
  "fcvtas v1.4s, v22.4s\n"  /*  cvt to int */                            \
  "fcvtas v2.4s, v23.4s\n"  /*  cvt to int */                            \
  "fcvtas v3.4s, v24.4s\n"  /*  cvt to int */                            \
  "sqxtn  v16.4h, v0.4s\n"  /*  cvt int32 to int16 */                    \
  "sqxtn2 v16.8h, v1.4s\n"  /*  cvt int32 to int16 */                    \
  "sqxtn  v17.4h, v2.4s\n"  /*  cvt int32 to int16 */                    \
  "sqxtn2 v17.8h, v3.4s\n"  /*  cvt int32 to int16 */                    \
  "sqxtn  v21.8b, v16.8h\n" /*  cvt int16 to int8 */                     \
  "sqxtn2 v21.16b, v17.8h\n" /*  cvt int16 to int8 */ /* store result */ \
  "str   q21,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_INT8_INT8_W8_V8_OUT                                             \
  SPARSE_INT8_INT8_W8_V8_KERNEL                                                \
  SPARSE_INT8_INT8_W8_V8_RELU                                                  \
  SPARSE_INT8_INT8_W8_V8_RELU6                                                 \
  SPARSE_INT8_INT8_W8_V8_LEAKY_RELU                                            \
  "ld1    {v8.4s},   [%[vmax]] \n" /* v8 = -127 */ /* data >= -127 */          \
  "fcmge v0.4s, v21.4s, v8.4s\n"                                               \
  "fcmge v1.4s, v22.4s, v8.4s\n" /* choose data */                             \
  "bif v21.16b,  v8.16b, v0.16b            \n"                                 \
  "bif v22.16b, v8.16b, v1.16b            \n"                                  \
  "fcvtas v0.4s, v21.4s\n"                           /*  cvt to int */         \
  "fcvtas v1.4s, v22.4s\n"                           /*  cvt to int */         \
  "sqxtn  v16.4h, v0.4s\n"                           /*  cvt int32 to int16 */ \
  "sqxtn2 v16.8h, v1.4s\n"                           /*  cvt int32 to int16 */ \
  "sqxtn  v21.8b, v16.8h\n" /*  cvt int16 to int8 */ /* store result */        \
  "str   d21,  [%[c_ptr]]\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_INT8_INT8_W4_V8_OUT                                            \
  SPARSE_INT8_INT8_W4_V8_KERNEL                                               \
  SPARSE_INT8_INT8_W4_V8_RELU                                                 \
  SPARSE_INT8_INT8_W4_V8_RELU6                                                \
  SPARSE_INT8_INT8_W4_V8_LEAKY_RELU                                           \
  "ld1    {v8.4s},   [%[vmax]]  \n" /* v8 = -127 */ /* data >= -127 */        \
  "fcmge v0.4s, v21.4s, v8.4s   \n"                 /* choose data */         \
  "bif v21.16b,  v8.16b, v0.16b \n"                                           \
  "fcvtas v0.4s, v21.4s\n"                          /*  cvt to int */         \
  "sqxtn  v16.4h, v0.4s\n"                          /*  cvt int32 to int16 */ \
  "sqxtn  v21.8b, v16.8h\n" /* cvt int16 to int8 */ /* store result */        \
  "str   s21,  [%[c_ptr]]\n"

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are int8.
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
void sparse_conv_int8_int8_pipelined(const int8_t* A,
                                     const int8_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float* bias,
                                     const float* scale,
                                     int8_t* output,
                                     int M,
                                     int K,
                                     int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  int flag_bias = (bias != nullptr) ? 1 : 0;
  size_t mc = N * sizeof(int8_t);
  size_t nc = M;
  size_t output_stride = N * sizeof(int8_t);
  size_t output_decrement = output_stride * nc - 48 * sizeof(int8_t);
  float vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  while
    SPARSE_LIKELY(mc >= 48 * sizeof(int8_t)) {
      const int8_t* w = A;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      const float* sc = scale;

      for (size_t i = 0; i < nc; i++) {
        uint32_t nnz = *nnzmap++;
        float vsclae = *sc++;
        float valpha = alpha;
        float vbias = (bias != nullptr) ? bias[i] : 0.0;
        // clang-format off
          asm volatile(SPARSE_INT8_INT8_W48_V8_OUT
            : [a_ptr] "+r"(w),
              [b_ptr] "+r"(B),
              [c_ptr] "+r"(output),
              [k] "+r"(nnz),
              [widx_dmap] "+r"(dmap)
            : [vscale] "r"(vsclae),
              [vbias] "r"(vbias),
              [vflag_act] "r"(flag_act),
              [valpha] "r"(valpha),
              [vmax] "r"(vmax)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w0", "w1", "x1", "cc", "memory");
        // clang-format on
        output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
      }
      output = reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
      B += 48;
      mc -= 48 * sizeof(int8_t);
    }
  if
    SPARSE_UNLIKELY(mc != 0) {
      output_decrement += 16 * sizeof(int8_t);
      if (mc & (32 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_INT8_W32_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha),
                [vmax] "r"(vmax)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v21", "v22", "v23", "v24", "v25", 
                "v26", "v27", "v28", "v30", "v31", "w0", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
        }
        output =
            reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
        B += 32;
        mc -= 32 * sizeof(int8_t);
      }
      output_decrement += 16 * sizeof(int8_t);
      if (mc & (16 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_INT8_W16_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha),
                [vmax] "r"(vmax)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v9", "v11", "v12", "v13", "v14", "v16", "v17", "v21", "v22", "v23",
                  "v24", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
        }
        output =
            reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
        B += 16;
        mc -= 16 * sizeof(int8_t);
      }
      output_decrement += 8 * sizeof(int8_t);
      if (mc & (8 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_INT8_W8_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha),
                [vmax] "r"(vmax)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12", "v16", "v21", 
              "v22", "w1", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
        }
        output =
            reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
        B += 8;
        mc -= 8 * sizeof(int8_t);
      }
      output_decrement += 4 * sizeof(int8_t);
      if (mc & (4 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_INT8_INT8_W4_V8_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha),
                [vmax] "r"(vmax)
              : "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v11", "v16", "v21", 
              "w1", "w2", "w3", "w4", "w5", "x1", "cc", "memory");
          // clang-format on
          output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
        }
        output =
            reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
        B += 4;
        mc -= 4 * sizeof(int8_t);
      }

      if
        SPARSE_UNLIKELY(mc != 0 && mc < 4 * sizeof(int8_t)) {
          const int8_t* w = A;
          const int32_t* dmap = widx_dmap;
          const uint32_t* nnzmap = nidx_nnzmap;
          const float* bs = bias;
          const float* sc = scale;
          // const float* al = alpha;
          float val = alpha;

          for (size_t i = 0; i < nc; i++) {
            float vbias = (bias != nullptr) ? *bs++ : 0;
            float vscale = *sc++;
            std::vector<float> out(mc, 0);
            uint32_t nnz = *nnzmap++;
            for (size_t j = 0; j < nnz; j++) {
              for (size_t k = 0; k < mc; k++) {
                out[k] += (*w) * (*(B + k));
              }
              w += 1;
              intptr_t diff = *dmap++;
              B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
            }
            for (size_t k = 0; k < mc; k++) {
              out[k] = out[k] * vscale + vbias;
              switch (flag_act) {
                case 0:
                  break;
                case 1:  // relu
                  out[k] = out[k] > 0 ? out[k] : 0;
                  break;
                case 2:  // relu6
                  out[k] = out[k] > 0 ? out[k] : 0;
                  out[k] = out[k] < val ? out[k] : val;
                  break;
                default:  // leaky_relu
                  out[k] = out[k] >= 0 ? out[k] : out[k] * val;
                  break;
              }
              float vax = out[k] > -127.0 ? out[k] : -127.0;
              vax = vax >= 0 ? vax + 0.5 : vax - 0.5;
              int32_t out_val = static_cast<int32_t>(vax);
              *(output + k) = out_val > 127 ? 127 : out_val;
            }
            output =
                reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
          }
        }
    }
}

#else  // armv7

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are f32.
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
void sparse_conv_fp32_pipelined(const float* A,
                                const float* B,
                                const int32_t* widx_dmap,
                                const uint32_t* nidx_nnzmap,
                                const float* bias,
                                float* output,
                                const int M,
                                const int K,
                                const int N,
                                const operators::SparseConvParam& param,
                                ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  return;
}

/**
 * \brief Sparse calculation implementation of 1x1 convolution, the input-output
 * type is int8-f32.
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
void sparse_conv_int8_fp32_pipelined(const int8_t* A,
                                     const int8_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float* bias,
                                     const float* scale,
                                     float* output,
                                     int M,
                                     int K,
                                     int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  return;
}

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are int8.
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
void sparse_conv_int8_int8_pipelined(const int8_t* A,
                                     const int8_t* B,
                                     const int32_t* widx_dmap,
                                     const uint32_t* nidx_nnzmap,
                                     const float* bias,
                                     const float* scale,
                                     int8_t* output,
                                     int M,
                                     int K,
                                     int N,
                                     const operators::SparseConvParam& param,
                                     ARMContext* ctx) {
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  return;
}

#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
