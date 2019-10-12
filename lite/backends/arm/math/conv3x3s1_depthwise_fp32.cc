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

#include <arm_neon.h>
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
#ifdef __aarch64__

#define INIT_PROCESS \
    "ld1 {v4.4s}, [%[inr0]], #16  \n" /* load input r0*/ \
    "ld1 {v6.4s}, [%[inr1]], #16  \n" /* load input r0*/ \
    "ld1 {v8.4s}, [%[inr2]], #16  \n" /* load input r0*/ \
    "ld1 {v10.4s}, [%[inr3]], #16  \n" /* load input r0*/ \
    "ld1 {v5.4s}, [%[inr0]]  \n" /* load input r0*/ \
    \
    /*out*/ \
    "ld1    {v20.4s},   [%[bias]] \n" /* load input r0*/ \
    "ld1    {v21.4s},   [%[bias]] \n" /* load input r0*/ \
    "ld1    {v22.4s},   [%[bias]] \n" /* load input r0*/ \
    "ld1    {v23.4s},   [%[bias]] \n" /* load input r0*/ \
    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v17 = 2345 */ \
    "ld1 {v7.4s}, [%[inr1]]  \n" /* load input r0*/ \
    "ld1 {v9.4s}, [%[inr2]]  \n" /* load input r0*/ \
    "ld1 {v11.4s}, [%[inr3]]  \n" /* load input r0*/ \

#define COMPUTE_PROCESS \
    "fmla v20.4s, v4.4s, %[w0].s[0] \n" /* din0123 * w0_0 */ \
    \
    "ld1 {v12.4s}, [%[inr4]], #16  \n" /* load input r0*/ \
    "ld1 {v14.4s}, [%[inr5]], #16  \n" /* load input r0*/ \
    "ld1 {v4.4s}, [%[inr0]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
    "ld1 {v13.4s}, [%[inr4]]  \n" /* load input r0*/ \
    /* r1 */ \
    "fmla v20.4s, v16.4s, %[w0].s[1] \n" /* din1234 * w0_1 */ \
    "ext  v18.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v19.16b, v6.16b, v7.16b, #8 \n" /* v17 = 2345 */ \
    "ld1 {v5.4s}, [%[inr0]]   \n" /*vld1q_f32(din_ptr0)*/ \
    "ld1 {v15.4s}, [%[inr5]]  \n" /* load input r0*/ \
    \
    "fmla v20.4s, v17.4s, %[w0].s[2] \n" /* din2345 * w0_2 */ \
    /* r1 */ \
    "fmla v21.4s, v6.4s, %[w0].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v20.4s, v6.4s, %[w1].s[0] \n" /* din0123 * w0_0 */ \
    \
    "ext  v16.16b, v8.16b, v9.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v17 = 2345 */ \
    "ld1 {v6.4s}, [%[inr1]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
    \
    "fmla v21.4s, v18.4s, %[w0].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v20.4s, v18.4s, %[w1].s[1] \n" /* din0123 * w0_0 */ \
    \
    "ext  v24.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v25.16b, v10.16b, v11.16b, #8 \n" /* v17 = 2345 */ \
    "ld1 {v7.4s}, [%[inr1]]   \n" /*vld1q_f32(din_ptr0)*/ \
    \
    "fmla v21.4s, v19.4s, %[w0].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v20.4s, v19.4s, %[w1].s[2] \n" /* din0123 * w0_0 */ \
    /* r2 */ \
    "fmla v22.4s, v8.4s, %[w0].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v21.4s, v8.4s, %[w1].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v20.4s, v8.4s, %[w2].s[0] \n" /* din0123 * w0_0 */ \
    "ld1 {v8.4s}, [%[inr2]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
    "ext  v18.16b, v12.16b, v13.16b, #4 \n" /* v16 = 1234 */ \
    \
    "fmla v22.4s, v16.4s, %[w0].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v21.4s, v16.4s, %[w1].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v20.4s, v16.4s, %[w2].s[1] \n" /* din0123 * w0_0 */ \
    \
    "ext  v19.16b, v12.16b, v13.16b, #8 \n" /* v17 = 2345 */ \
    "ld1 {v9.4s}, [%[inr2]]   \n" /*vld1q_f32(din_ptr0)*/ \
    \
    "fmla v22.4s, v17.4s, %[w0].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v21.4s, v17.4s, %[w1].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v20.4s, v17.4s, %[w2].s[2] \n" /* din0123 * w0_0 */ \
    \
    /* r3 */ \
    "fmla v23.4s, v10.4s, %[w0].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v10.4s, %[w1].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v21.4s, v10.4s, %[w2].s[0] \n" /* din0123 * w0_0 */ \
    "ld1 {v10.4s}, [%[inr3]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
    "ext  v16.16b, v12.16b, v13.16b, #4 \n" /* v16 = 1234 */ \
    \
    "fmla v23.4s, v24.4s, %[w0].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v24.4s, %[w1].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v21.4s, v24.4s, %[w2].s[1] \n" /* din0123 * w0_0 */ \
    \
    "ext  v17.16b, v12.16b, v13.16b, #8 \n" /* v17 = 2345 */ \
    "ld1 {v11.4s}, [%[inr3]]   \n" /*vld1q_f32(din_ptr0)*/ \
    \
    "fmla v23.4s, v25.4s, %[w0].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v25.4s, %[w1].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v21.4s, v25.4s, %[w2].s[2] \n" /* din0123 * w0_0 */ \

#define RESULT_PROCESS \
    /* r4 */ \
    "fmla v23.4s, v12.4s, %[w1].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v12.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "str    q20, [%[outc00]], #16\n" /* save outc00*/ \
    "str    q21, [%[outc01]], #16\n" /* save outc00*/ \
    \
    "fmla v23.4s, v16.4s, %[w1].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v16.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "ld1    {v20.4s},   [%[bias]] \n" /* load input r0*/ \
    "ld1    {v21.4s},   [%[bias]] \n" /* load input r0*/ \
    \
    "ext  v18.16b, v14.16b, v15.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v19.16b, v14.16b, v15.16b, #8 \n" /* v17 = 2345 */ \
    \
    "fmla v23.4s, v17.4s, %[w1].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v17.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    \
    /* r5 */ \
    "fmla v23.4s, v14.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v17 = 2345 */ \
    \
    "fmla v23.4s, v18.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "str    q22, [%[outc02]], #16\n" /* save outc00*/ \
    "ld1    {v22.4s},   [%[bias]] \n" /* load input r0*/ \
    \
    "fmla v23.4s, v19.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    \
    "str    q23, [%[outc03]], #16\n" /* save outc00*/ \
    "ld1    {v23.4s},   [%[bias]] \n" /* load input r0*/ \

#define RESULT_REMAIN_PROCESS \
    /* r4 */ \
    "fmla v23.4s, v12.4s, %[w1].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v12.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "ld1 {v4.4s}, [%[outc00]]  \n" \
    "ld1 {v5.4s}, [%[outc01]]  \n" \
    \
    "fmla v23.4s, v16.4s, %[w1].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v16.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "bif  v20.16b, v4.16b, %[mask].16b    \n" /*pipei*/ \
    "bif  v21.16b, v5.16b, %[mask].16b    \n" /*pipei*/ \
    \
    "ext  v18.16b, v14.16b, v15.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v19.16b, v14.16b, v15.16b, #8 \n" /* v17 = 2345 */ \
    \
    "fmla v23.4s, v17.4s, %[w1].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v17.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    "str    q20, [%[outc00]], #16\n" /* save outc00*/ \
    "str    q21, [%[outc01]], #16\n" /* save outc00*/ \
    \
    /* r5 */ \
    "fmla v23.4s, v14.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "ld1 {v4.4s}, [%[outc02]]  \n" \
    "ld1 {v5.4s}, [%[outc03]]  \n" \
    \
    "fmla v23.4s, v18.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "bif  v22.16b, v4.16b, %[mask].16b    \n" /*pipei*/ \
    \
    "fmla v23.4s, v19.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    \
    "bif  v23.16b, v5.16b, %[mask].16b    \n" /*pipei*/ \
    "str    q22, [%[outc02]], #16\n" /* save outc00*/ \
    "str    q23, [%[outc03]], #16\n" /* save outc00*/ \

#define RESULT_RELU_PROCESS \
    /* r4 */ \
    "movi v31.4s, #0 \n" \
    "fmla v23.4s, v12.4s, %[w1].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v12.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "fmax v20.4s, v20.4s, v31.4s\n" \
    "fmax v21.4s, v21.4s, v31.4s\n" \
    \
    "fmla v23.4s, v16.4s, %[w1].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v16.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "str    q20, [%[outc00]], #16\n" /* save outc00*/ \
    "str    q21, [%[outc01]], #16\n" /* save outc00*/ \
    \
    "ext  v18.16b, v14.16b, v15.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v19.16b, v14.16b, v15.16b, #8 \n" /* v17 = 2345 */ \
    \
    "fmla v23.4s, v17.4s, %[w1].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v17.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    "ld1    {v20.4s},   [%[bias]] \n" /* load input r0*/ \
    "ld1    {v21.4s},   [%[bias]] \n" /* load input r0*/ \
    \
    /* r5 */ \
    "fmla v23.4s, v14.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v17 = 2345 */ \
    "fmax v22.4s, v22.4s, v31.4s\n" \
    \
    "fmla v23.4s, v18.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "str    q22, [%[outc02]], #16\n" /* save outc00*/ \
    "ld1    {v22.4s},   [%[bias]] \n" /* load input r0*/ \
    \
    "fmla v23.4s, v19.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    \
    "fmax v23.4s, v23.4s, v31.4s\n" \
    "str    q23, [%[outc03]], #16\n" /* save outc00*/ \
    "ld1    {v23.4s},   [%[bias]] \n" /* load input r0*/ \

#define RESULT_REMAIN_RELU_PROCESS \
    /* r4 */ \
    "movi v31.4s, #0 \n" \
    "fmla v23.4s, v12.4s, %[w1].s[0] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v12.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "ld1 {v4.4s}, [%[outc00]]  \n" \
    "ld1 {v5.4s}, [%[outc01]]  \n" \
    "fmax v20.4s, v20.4s, v31.4s\n" \
    "fmax v21.4s, v21.4s, v31.4s\n" \
    \
    "fmla v23.4s, v16.4s, %[w1].s[1] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v16.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "bif  v20.16b, v4.16b, %[mask].16b    \n" /*pipei*/ \
    "bif  v21.16b, v5.16b, %[mask].16b    \n" /*pipei*/ \
    \
    "ext  v18.16b, v14.16b, v15.16b, #4 \n" /* v16 = 1234 */ \
    "ext  v19.16b, v14.16b, v15.16b, #8 \n" /* v17 = 2345 */ \
    \
    "fmla v23.4s, v17.4s, %[w1].s[2] \n" /* din1234 * w0_0 */ \
    "fmla v22.4s, v17.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    "str    q20, [%[outc00]], #16\n" /* save outc00*/ \
    "str    q21, [%[outc01]], #16\n" /* save outc00*/ \
    \
    /* r5 */ \
    "fmla v23.4s, v14.4s, %[w2].s[0] \n" /* din1234 * w0_0 */ \
    \
    "ld1 {v4.4s}, [%[outc02]]  \n" \
    "ld1 {v5.4s}, [%[outc03]]  \n" \
    "fmax v22.4s, v22.4s, v31.4s\n" \
    \
    "fmla v23.4s, v18.4s, %[w2].s[1] \n" /* din1234 * w0_0 */ \
    \
    "bif  v22.16b, v4.16b, %[mask].16b    \n" /*pipei*/ \
    \
    "fmla v23.4s, v19.4s, %[w2].s[2] \n" /* din1234 * w0_0 */ \
    \
    "fmax v23.4s, v23.4s, v31.4s\n" \
    "str    q22, [%[outc02]], #16\n" /* save outc00*/ \
    "bif  v23.16b, v5.16b, %[mask].16b    \n" /*pipei*/ \
    "str    q23, [%[outc03]], #16\n" /* save outc00*/ \

#else
#define INIT_PROCESS \
    "vld1.32    {d8-d9}, [%[inr0]]!  \n" /*     @ load din0_0123, to q4 */ \
    "vld1.32    {d12-d13}, [%[inr1]]!   \n" /*    @ load din1_0123, to q6 */ \
    "vld1.32    {d16-d17}, [%[inr2]]!    \n" /*   @ load din2_0123, to q8 */ \
    "vld1.32    {d20-d21}, [%[inr3]]!    \n" /*   @ load din3_0123, to q10 */ \
    "vld1.32    {d10}, [%[inr0]]      \n" /* @ load din0_45, to d10 */ \
    "vld1.32    {d14}, [%[inr1]]      \n" /* @ load din1_45, to d10 */ \
    "vld1.32    {d18}, [%[inr2]]      \n" /* @ load din2_45, to d10 */ \
    "vld1.32    {d22}, [%[inr3]]      \n" /* @ load din3_45, to d10 */ \
    /*out*/ \
    "vld1.32    {d28-d29},   [%[bias]] \n" /* load output r0*/ \
    "vld1.32    {d30-d31},   [%[bias]] \n" /* load output r1*/ \
    "vext.32  q12, q4, q5, #1     \n" /* v16 = 1234 */ \
    "vext.32  q13, q4, q5, #2    \n"/* v16 = 2345 */ \

#define COMPUTE_PROCESS \
    /*r0 */ \
    "vmla.f32 q14, q4, %e[w0][0] \n" /* din0123 * w0_0  */ \
    "vld1.32    {d8-d9}, [%[inr0]]!     \n" /* @ load din1_2345, to q6 */\
    "vmla.f32 q14, q12, %e[w0][1] \n" /* din1234 * w0_1  */ \
    "vext.32  q12, q6, q7, #1     \n" /* v16 = 1234 */ \
    "vld1.32    {d10}, [%[inr0]]      \n" /* @ load din0_45, to d10 */ \
    "vmla.f32 q14, q13, %f[w0][0] \n" /* din2345 * w0_2  */ \
    "vext.32  q13, q6, q7, #2    \n"/* v16 = 2345 */ \
    /*r1 */ \
    "vmla.f32 q15, q6, %e[w0][0] \n" /* din0123 * w0_0  */ \
    "vmla.f32 q14, q6, %e[w1][0] \n" /* din0123 * w0_0  */ \
    "vld1.32    {d12-d13}, [%[inr1]]!      \n" \
    "vmla.f32 q15, q12, %e[w0][1] \n" /* din1234 * w0_1  */ \
    "vmla.f32 q14, q12, %e[w1][1] \n" /* din1234 * w0_1  */ \
    "vld1.32    {d14}, [%[inr1]]      \n" /* @ load din1_45, to d10 */ \
    "vext.32  q12, q8, q9, #1     \n" /* v16 = 1234 */ \
    "vmla.f32 q15, q13, %f[w0][0] \n" /* din2345 * w0_2  */ \
    "vmla.f32 q14, q13, %f[w1][0] \n" /* din2345 * w0_2  */ \
    "vext.32  q13, q8, q9, #2    \n"/* v16 = 2345 */ \
    /*r2 */ \
    "vmla.f32 q15, q8, %e[w1][0] \n" /* din0123 * w0_0  */ \
    "vmla.f32 q14, q8, %e[w2][0] \n" /* din0123 * w0_0  */ \
    "vld1.32    {d16-d17}, [%[inr2]]!    \n" /*   @ load din2_2345, to q8 */ \
    "vmla.f32 q15, q12, %e[w1][1] \n" /* din1234 * w0_1  */ \
    "vmla.f32 q14, q12, %e[w2][1] \n" /* din1234 * w0_1  */ \
    "vld1.32    {d18}, [%[inr2]]    \n" /*   @ load din2_2345, to q8 */ \
    "vmla.f32 q15, q13, %f[w1][0] \n" /* din2345 * w0_2  */ \
    "vmla.f32 q14, q13, %f[w2][0] \n" /* din2345 * w0_2  */ \
    "vext.32  q12, q10, q11, #1     \n" /* v16 = 1234 */ \
    "vext.32  q13, q10, q11, #2    \n"/* v16 = 2345 */ \
    /*r3 */ \
    "vmla.f32 q15, q10, %e[w2][0] \n" /* din0123 * w0_0  */ \
    "vld1.32    {d20-d21}, [%[inr3]]!      \n" /* @ load din3_2345, to q10 */ \
    "vmla.f32 q15, q12, %e[w2][1] \n" /* din1234 * w0_1  */ \
    "vld1.32    {d22}, [%[inr3]]      \n" /* @ load din3_2345, to q10 */ \
    "vmla.f32 q15, q13, %f[w2][0] \n" /* din2345 * w0_2  */ \

#define RESULT_PROCESS \
    "vext.32  q12, q4, q5, #1     \n" /* v16 = 1234 */ \
    "vext.32  q13, q4, q5, #2    \n"/* v16 = 2345 */ \
    "vst1.32  {d28-d29}, [%[outc00]]! \n" /* store*/ \
    "vst1.32  {d30-d31}, [%[outc01]]! \n" /* store*/ \
    /*out*/ \
    "vld1.32    {d28-d29},   [%[bias]] \n" /* load output r0*/ \
    "vld1.32    {d30-d31},   [%[bias]] \n" /* load output r1*/ \

#define RESULT_REMAIN_PROCESS \
    "vld1.32    {d8-d9}, [%[outc00]]      \n" /* @ load din3_2345, to q10 */ \
    "vld1.32    {d10-d11}, [%[outc01]]      \n" /* @ load din3_2345, to q10 */ \
    "vbif d28, d8, %e[mask] \n" /* bif*/ \
    "vbif d29, d9, %f[mask] \n" /* bif*/ \
    "vbif d30, d10, %e[mask] \n" /* bif*/ \
    "vbif d31, d11, %f[mask] \n" /* bif*/ \
    "vst1.32  {d28-d29}, [%[outc00]]! \n" /* store*/ \
    "vst1.32  {d30-d31}, [%[outc01]]! \n" /* store*/ \

#define RESULT_RELU_PROCESS \
    "vmov.u32 d11, #0 \n" /* 0*/ \
    "vext.32  q12, q4, q5, #1     \n" /* v16 = 1234 */ \
    "vext.32  q13, q4, q5, #2    \n"/* v16 = 2345 */ \
    "vmax.f32 d28, d28, d11 \n" /* for relu */ \
    "vmax.f32 d29, d29, d11 \n" /* for relu */ \
    "vmax.f32 d30, d30, d11 \n" /* for relu */ \
    "vmax.f32 d31, d31, d11 \n" /* for relu */ \
    "vst1.32  {d28-d29}, [%[outc00]]! \n" /* store*/ \
    "vst1.32  {d30-d31}, [%[outc01]]! \n" /* store*/ \
    /*out*/ \
    "vld1.32    {d28-d29},   [%[bias]] \n" /* load output r0*/ \
    "vld1.32    {d30-d31},   [%[bias]] \n" /* load output r1*/ \

#define RESULT_REMAIN_RELU_PROCESS \
    "vmov.u32 q6, #0 \n" /* 0*/ \
    "vld1.32    {d8-d9}, [%[outc00]]      \n" /* @ load din3_2345, to q10 */ \
    "vld1.32    {d10-d11}, [%[outc01]]      \n" /* @ load din3_2345, to q10 */ \
    "vmax.f32 q14, q14, q6 \n" /* for relu */ \
    "vmax.f32 q15, q15, q6 \n" /* for relu */ \
    "vbif d28, d8, %e[mask] \n" /* bif*/ \
    "vbif d29, d9, %f[mask] \n" /* bif*/ \
    "vbif d30, d10, %e[mask] \n" /* bif*/ \
    "vbif d31, d11, %f[mask] \n" /* bif*/ \
    "vst1.32  {d28-d29}, [%[outc00]]! \n" /* store*/ \
    "vst1.32  {d30-d31}, [%[outc01]]! \n" /* store*/ \

#endif

void compute_process(const float* inr0, const float* inr1, const float* inr2, \
                          const float* inr3, const float* inr4, const float* inr5,
                          int cnt, int remain, float* bias, float32x4_t w0, float32x4_t w1, \
                          float32x4_t w2, uint32x4_t mask,
                          float* outc00, float* outc01, float* outc02, float* outc03){
#ifdef __aarch64__
  asm volatile(
    INIT_PROCESS
    "cmp %w[cnt], #1                           \n"
    "blt 2f                                     \n"
    "0: \n"
        COMPUTE_PROCESS
        RESULT_PROCESS
        "subs %w[cnt], %w[cnt], #1 \n"
        "bne 0b \n"
    "2: \n"
        "cmp %w[remain], #1                           \n"
        "blt 1f                                     \n"
        COMPUTE_PROCESS
        RESULT_REMAIN_PROCESS
    "1: \n"
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [inr4] "+r"(inr4), [inr5] "+r"(inr5),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [outc02] "+r"(outc02), [outc03] "+r"(outc03),
      [cnt] "+r" (cnt)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [mask] "w"(mask),
        [bias] "r" (bias), [remain] "r" (remain)
    : "cc", "memory",
      "v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16","v17","v18","v19","v20","v21","v22", "v23", "v24", "v25", "v31"
    );
#else
  asm volatile(
    INIT_PROCESS
    "cmp %[cnt], #1                           \n"
    "blt 2f                                     \n"
    "0: \n"
        COMPUTE_PROCESS
        RESULT_PROCESS
        "subs %[cnt], %[cnt], #1 \n"
        "bne 0b \n"
    "2: \n"
        "cmp %[remain], #1                           \n"
        "blt 1f                                     \n"
        COMPUTE_PROCESS
        RESULT_REMAIN_PROCESS
    "1: \n"
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [cnt] "+r" (cnt)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [mask] "w"(mask),
        [bias] "r" (bias), [remain] "r" (remain)
    : "cc", "memory",
      "q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
    );
#endif
}

void compute_s_process(const float* inr0, const float* inr1, const float* inr2, \
                          const float* inr3, const float* inr4, const float* inr5,
                          float* bias, float32x4_t w0, float32x4_t w1, \
                          float32x4_t w2, /* uint32x4_t mask, */ \
                          float* outc00, float* outc01, float* outc02, float* outc03){
#ifdef __aarch64__
  asm volatile(
    INIT_PROCESS
    COMPUTE_PROCESS
    RESULT_PROCESS
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [inr4] "+r"(inr4), [inr5] "+r"(inr5),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [outc02] "+r"(outc02), [outc03] "+r"(outc03)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [bias] "r" (bias)
    : "cc", "memory",
      "v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16","v17","v18","v19","v20","v21","v22", "v23", "v24", "v25", "v31"
    );
#else
  asm volatile(
    INIT_PROCESS
    COMPUTE_PROCESS
    RESULT_PROCESS
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [outc02] "+r"(outc02), [outc03] "+r"(outc03)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [bias] "r" (bias)
    : "cc", "memory",
      "q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
    );
#endif
}

void compute_relu_process(const float* inr0, const float* inr1, const float* inr2, \
                          const float* inr3, const float* inr4, const float* inr5,
                          int cnt, int remain, float* bias, float32x4_t w0, float32x4_t w1, \
                          float32x4_t w2, uint32x4_t mask,
                          float* outc00, float* outc01, float* outc02, float* outc03){
#ifdef __aarch64__
  asm volatile(
    INIT_PROCESS
    "cmp %w[cnt], #1                           \n"
    "blt 2f                                     \n"
    "0: \n"
        COMPUTE_PROCESS
        RESULT_RELU_PROCESS
        "subs %w[cnt], %w[cnt], #1 \n"
        "bne 0b \n"
    "2: \n"
        "cmp %w[remain], #1                           \n"
        "blt 1f                                     \n"
        COMPUTE_PROCESS
        RESULT_REMAIN_RELU_PROCESS
    "1: \n"
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [inr4] "+r"(inr4), [inr5] "+r"(inr5),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [outc02] "+r"(outc02), [outc03] "+r"(outc03),
      [cnt] "+r" (cnt)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [mask] "w"(mask),
        [bias] "r" (bias), [remain] "r" (remain)
    : "cc", "memory",
      "v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16","v17","v18","v19","v20","v21","v22", "v23", "v24", "v25", "v31"
    );
#else
  asm volatile(
    INIT_PROCESS
    "cmp %[cnt], #1                           \n"
    "blt 2f                                     \n"
    "0: \n"
        COMPUTE_PROCESS
        RESULT_RELU_PROCESS
        "subs %[cnt], %[cnt], #1 \n"
        "bne 0b \n"
    "2: \n"
        "cmp %[remain], #1                           \n"
        "blt 1f                                     \n"
        COMPUTE_PROCESS
        RESULT_REMAIN_RELU_PROCESS
    "1: \n"
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [cnt] "+r" (cnt)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [mask] "w"(mask),
        [bias] "r" (bias), [remain] "r" (remain)
    : "cc", "memory",
      "q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
    );

#endif
}
void compute_s_relu_process(const float* inr0, const float* inr1, const float* inr2, \
                          const float* inr3, const float* inr4, const float* inr5,
                          float* bias, float32x4_t w0, float32x4_t w1, \
                          float32x4_t w2, /* uint32x4_t mask,*/ \
                          float* outc00, float* outc01, float* outc02, float* outc03){
#ifdef __aarch64__
  asm volatile(
    INIT_PROCESS
    COMPUTE_PROCESS
    RESULT_RELU_PROCESS
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [inr4] "+r"(inr4), [inr5] "+r"(inr5),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01),
      [outc02] "+r"(outc02), [outc03] "+r"(outc03)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [bias] "r" (bias)
    : "cc", "memory",
      "v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16","v17","v18","v19","v20","v21","v22", "v23", "v24", "v25", "v31"
    );
#else
  asm volatile(
    INIT_PROCESS
    COMPUTE_PROCESS
    RESULT_RELU_PROCESS
    :[inr0] "+r"(inr0), [inr1] "+r"(inr1),
      [inr2] "+r"(inr2), [inr3] "+r"(inr3),
      [outc00] "+r"(outc00), [outc01] "+r"(outc01)
    :[w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [bias] "r" (bias)
    : "cc", "memory",
      "q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
    );
#endif
}

void prepack_input_dw(const float *din_batch, float *pre_din, int c, int hs, int he, int ws, int we, \
                      int size_in_channel, int ih, int win, float *ptr_zero){
  const float* din = din_batch + c * size_in_channel;
  int w_len = we - ws;
  for (int i = hs; i < he; i++){
    if (i < 0 || i >= ih){
        memcpy(pre_din, ptr_zero, w_len * sizeof(float));
        pre_din += w_len;
        continue;
    }else{
        for (int j = ws; j < 0; j++){
            *pre_din++ = 0.f;
        }
        memcpy(pre_din, din, win * sizeof(float));
        pre_din += win;
        din += win;
        for (int j = win; j < we; j++){
            *pre_din++ = 0.f;
        }
    }
  }
}

void conv_3x3s1_depthwise_s_fp32(const float* i_data,
                               float* o_data,
                               int bs,
                               int oc,
                               int oh,
                               int ow,
                               int ic,
                               int ih,
                               int win,
                               const float* weights,
                               const float* bias,
                               int pad_w,
                               int pad_h,
                               bool flag_bias,
                               bool flag_relu,
                               ARMContext* ctx){
    int threads = ctx->threads();
#ifdef __aarch64__
  const int out_h_kernel = 4;
  const int out_w_kernel = 4;
#else
  const int out_h_kernel = 2;
  const int out_w_kernel = 4;
#endif
  const int win_ext = win + 2 * pad_w;
  const int hin_ext = ih + 2 * pad_h;
  // const int tile_w = ow >> 2;
  // const int remain = ow % 4;
  const int prein_size = win_ext * hin_ext;
  // auto workspace_size = threads * prein_size;
  // ctx->ExtendWorkspace(sizeof(float) * workspace_size);


  // get workspace
  float* ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_ext);
  float* ptr_write = ptr_zero + win_ext;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = win + pad_w;
  int hs = -pad_h;
  int he = ih + pad_h;

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c++) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow;
#endif
      /// const array size
      prepack_input_dw(
          din_batch, pre_din, c, hs, he, ws, we, size_in_channel, ih, win, ptr_zero);
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;

      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c];
        bias_local[2] = bias[c];
        bias_local[3] = bias[c];
      }
      float32x4_t vbias = vld1q_f32(bias_local);
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 3);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 6);   // w2, v25

      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc00 = dout_c00 + h * ow;
        float* outc01 = outc00 + ow;
        float* outc02 = outc01 + ow;
        float* outc03 = outc02 + ow;

        const float* inr0 = pre_din + h * win_ext;
        const float* inr1 = inr0 + win_ext;
        const float* inr2 = inr1 + win_ext;
        const float* inr3 = inr2 + win_ext;
        const float* inr4 = inr3 + win_ext;
        const float* inr5 = inr4 + win_ext;
#ifdef __aarch64__
        //! process bottom pad
        if (h + 5 > hin_ext) {
          switch (h + 5 - hin_ext) {
            case 4:
              inr1 = ptr_zero;
            case 3:
              inr2 = ptr_zero;
            case 2:
              inr3 = ptr_zero;
            case 1:
              inr4 = ptr_zero;
            case 0:
              inr5 = ptr_zero;
            default:
              break;
          }
        }
        //! process bottom remain
        if (h + out_h_kernel > oh) {
          switch (h + out_h_kernel - oh) {
            case 3:
              outc01 = ptr_write;
            case 2:
              outc02 = ptr_write;
            case 1:
              outc03 = ptr_write;
            default:
              break;
          }
        }
#else
        //! process bottom pad
        if (h + 3 > hin_ext) {
          switch (h + 3 - hin_ext) {
            case 2:
              inr1 = ptr_zero;
            case 1:
              inr2 = ptr_zero;
            case 0:
              inr3 = ptr_zero;
            default:
              break;
          }
        }
        //! process bottom remain
        if (h + out_h_kernel > oh) {
          switch (h + out_h_kernel - oh) {
            case 1:
              outc01 = ptr_write;
            default:
              break;
          }
        }
#endif
        float out_buf0[4];
        float out_buf1[4];
        float out_buf2[4];
        float out_buf3[4];
        if (flag_relu){
            compute_s_relu_process(inr0, inr1, inr2, inr3, inr4, inr5, \
                        bias_local, w0, w1, w2, \
                        out_buf0, out_buf1, out_buf2, out_buf3);
        }else{
            compute_s_process(inr0, inr1, inr2, inr3, inr4, inr5, \
                        bias_local, w0, w1, w2, \
                        out_buf0, out_buf1, out_buf2, out_buf3);
        }
        for (int i = 0; i < ow; i++){
            *outc00++ = out_buf0[i];
            *outc01++ = out_buf1[i];
            *outc02++ = out_buf2[i];
            *outc03++ = out_buf3[i];
        }
      }
    }
  }
}

void print(float* ptr, int h, int w){
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++){
            printf("%3f ", *ptr++);
        }
        printf("\n");
  }
}

void conv_3x3s1_depthwise_fp32(const float* i_data,
                               float* o_data,
                               int bs,
                               int oc,
                               int oh,
                               int ow,
                               int ic,
                               int ih,
                               int win,
                               const float* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  // printf("conv_3x3s1_depthwise_fp32 new\n");
  const int pad_h = param.paddings[0];
  const int pad_w = param.paddings[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  if (ow < 4){
        conv_3x3s1_depthwise_s_fp32(i_data, o_data, bs, oc, oh, ow, ic, ih, win, weights, bias, \
                                    pad_w, pad_h, flag_bias, flag_relu, ctx);
        return;
    }
    int threads = ctx->threads();
#ifdef __aarch64__
  const int out_h_kernel = 4;
  const int out_w_kernel = 4;
#else
  const int out_h_kernel = 2;
  const int out_w_kernel = 4;
#endif
  const int win_ext = win + 2 * pad_w;
  const int hin_ext = ih + 2 * pad_h;
  const int tile_w = ow >> 2;
  const int remain = ow % 4;
  const int prein_size = win_ext * hin_ext;
  // auto workspace_size = threads * prein_size;
  // ctx->ExtendWorkspace(sizeof(float) * workspace_size);


  // get workspace
  float* ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_ext);
  float* ptr_write = ptr_zero + win_ext;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  // const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  // float bias_local[4] = {0, 0, 0, 0};
    //! for 4x6 convolution window
  const unsigned int right_pad_idx[4] = {0, 1, 2, 3};

  // uint32x4_t vmask_rp1 = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
  // uint32x4_t vmask_rp2 = vcgeq_u32(vld1q_u32(right_pad_idx + 4), vdupq_n_u32(size_pad_right));
  uint32x4_t mask = vcgtq_u32(vdupq_n_u32(remain), vld1q_u32(right_pad_idx));
  // unsigned int rmask[4];
  // vst1q_u32(rmask, vmask_result);

  int ws = -pad_w;
  int we = win + pad_w;
  int hs = -pad_h;
  int he = ih + pad_h;

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c++) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow;
#endif
      /// const array size
      prepack_input_dw(
          din_batch, pre_din, c, hs, he, ws, we, size_in_channel, ih, win, ptr_zero);
      // printf("prepack_input_dw \n");
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;

      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c];
        bias_local[2] = bias[c];
        bias_local[3] = bias[c];
      }
      float32x4_t vbias = vld1q_f32(bias_local);
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 3);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 6);   // w2, v25

      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc00 = dout_c00 + h * ow;
        float* outc01 = outc00 + ow;
        float* outc02 = outc01 + ow;
        float* outc03 = outc02 + ow;

        const float* inr0 = pre_din + h * win_ext;
        const float* inr1 = inr0 + win_ext;
        const float* inr2 = inr1 + win_ext;
        const float* inr3 = inr2 + win_ext;
        const float* inr4 = inr3 + win_ext;
        const float* inr5 = inr4 + win_ext;
#ifdef __aarch64__
        //! process bottom pad
        if (h + 5 > hin_ext) {
          switch (h + 5 - hin_ext) {
            case 4:
              inr1 = ptr_zero;
            case 3:
              inr2 = ptr_zero;
            case 2:
              inr3 = ptr_zero;
            case 1:
              inr4 = ptr_zero;
            case 0:
              inr5 = ptr_zero;
            default:
              break;
          }
        }
        //! process bottom remain
        if (h + out_h_kernel > oh) {
          switch (h + out_h_kernel - oh) {
            case 3:
              outc01 = ptr_write;
            case 2:
              outc02 = ptr_write;
            case 1:
              outc03 = ptr_write;
            default:
              break;
          }
        }
#else
        //! process bottom pad
        if (h + 3 > hin_ext) {
          switch (h + 3 - hin_ext) {
            case 2:
              inr1 = ptr_zero;
            case 1:
              inr2 = ptr_zero;
            case 0:
              inr3 = ptr_zero;
            default:
              break;
          }
        }
        //! process bottom remain
        if (h + out_h_kernel > oh) {
          switch (h + out_h_kernel - oh) {
            case 1:
              outc01 = ptr_write;
            default:
              break;
          }
        }
#endif
        int cnt = tile_w;
        if (flag_relu){
            // printf("compute_relu_process\n");
            compute_relu_process(inr0, inr1, inr2, inr3, inr4, inr5, \
                          cnt, remain, bias_local, w0, w1, w2, mask, \
                          outc00, outc01, outc02, outc03);
        }else{
            // printf("compute_process\n");
            compute_process(inr0, inr1, inr2, inr3, inr4, inr5, \
                          cnt, remain, bias_local, w0, w1, w2, mask, \
                          outc00, outc01, outc02, outc03);
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
