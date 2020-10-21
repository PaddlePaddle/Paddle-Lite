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
void conv_3x3s1_depthwise_fp32_bias(const float* i_data,
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
                                    float* relu_ptr,
                                    float* six_ptr,
                                    float* scale_ptr,
                                    const operators::ConvParam& param,
                                    ARMContext* ctx);

void conv_3x3s1_depthwise_fp32_relu(const float* i_data,
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
                                    float* relu_ptr,
                                    float* six_ptr,
                                    float* scale_ptr,
                                    const operators::ConvParam& param,
                                    ARMContext* ctx);

void conv_3x3s1_depthwise_fp32_relu6(const float* i_data,
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
                                     float* relu_ptr,
                                     float* six_ptr,
                                     float* scale_ptr,
                                     const operators::ConvParam& param,
                                     ARMContext* ctx);

void conv_3x3s1_depthwise_fp32_leakyRelu(const float* i_data,
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
                                         float* relu_ptr,
                                         float* six_ptr,
                                         float* scale_ptr,
                                         const operators::ConvParam& param,
                                         ARMContext* ctx);
// clang-format off
#ifdef __aarch64__
#define COMPUTE \
          "ldp    q0, q1,   [%[inr0]], #32\n" /* load input r0*/ \
          "ldp    q6, q7,   [%[inr1]], #32\n" /* load input r1*/ \
          "ldp    q2, q3,   [%[inr0]], #32\n" /* load input r0*/ \
          "ldp    q8, q9,   [%[inr1]], #32\n" /* load input r1*/ \
          "ldp    q4, q5,   [%[inr0]]\n"      /* load input r0*/ \
          "ldp    q10, q11, [%[inr1]]\n"      /* load input r1*/ \
          /*  r0, r1, mul w0, get out r0, r1 */ \
          "fmul   v15.4s ,  %[w0].4s,  v0.4s\n" /* outr00 = w0 * r0, 0*/ \
          "fmul   v16.4s ,  %[w0].4s,  v1.4s\n" /* outr01 = w0 * r0, 1*/ \
          "fmul   v17.4s ,  %[w0].4s,  v2.4s\n" /* outr02 = w0 * r0, 2*/ \
          "fmul   v18.4s ,  %[w0].4s,  v3.4s\n" /* outr03 = w0 * r0, 3*/ \
          "fmul   v19.4s ,  %[w0].4s,  v6.4s\n" /* outr10 = w0 * r1, 0*/ \
          "fmul   v20.4s ,  %[w0].4s,  v7.4s\n" /* outr11 = w0 * r1, 1*/ \
          "fmul   v21.4s ,  %[w0].4s,  v8.4s\n" /* outr12 = w0 * r1, 2*/ \
          "fmul   v22.4s ,  %[w0].4s,  v9.4s\n" /* outr13 = w0 * r1, 3*/ \
          /*  r0, r1, mul w1, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w1].4s,  v1.4s\n" /* outr00 = w1 * r0[1]*/ \
          "ldp    q0, q1,   [%[inr2]], #32\n"     /* load input r2*/ \
          "fmla   v16.4s ,  %[w1].4s,  v2.4s\n" /* outr01 = w1 * r0[2]*/ \
          "fmla   v17.4s ,  %[w1].4s,  v3.4s\n" /* outr02 = w1 * r0[3]*/ \
          "fmla   v18.4s ,  %[w1].4s,  v4.4s\n" /* outr03 = w1 * r0[4]*/ \
          "fmla   v19.4s ,  %[w1].4s,  v7.4s\n" /* outr10 = w1 * r1[1]*/ \
          "fmla   v20.4s ,  %[w1].4s,  v8.4s\n" /* outr11 = w1 * r1[2]*/ \
          "fmla   v21.4s ,  %[w1].4s,  v9.4s\n" /* outr12 = w1 * r1[3]*/ \
          "fmla   v22.4s ,  %[w1].4s,  v10.4s\n"/* outr13 = w1 * r1[4]*/ \
          /*  r0, r1, mul w2, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w2].4s,  v2.4s\n" /* outr00 = w2 * r0[2]*/ \
          "fmla   v16.4s ,  %[w2].4s,  v3.4s\n" /* outr01 = w2 * r0[3]*/ \
          "ldp    q2, q3,   [%[inr2]], #32\n"     /* load input r2*/ \
          "fmla   v17.4s ,  %[w2].4s,  v4.4s\n" /* outr02 = w2 * r0[4]*/ \
          "fmla   v18.4s ,  %[w2].4s,  v5.4s\n" /* outr03 = w2 * r0[5]*/ \
          "ldp    q4, q5,   [%[inr2]]\n"          /* load input r2*/ \
          "fmla   v19.4s ,  %[w2].4s,  v8.4s\n" /* outr10 = w2 * r1[2]*/ \
          "fmla   v20.4s ,  %[w2].4s,  v9.4s\n" /* outr11 = w2 * r1[3]*/ \
          "fmla   v21.4s ,  %[w2].4s,  v10.4s\n"/* outr12 = w2 * r1[4]*/ \
          "fmla   v22.4s ,  %[w2].4s,  v11.4s\n"/* outr13 = w2 * r1[5]*/ \
          /*  r1, r2, mul w3, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w3].4s,  v6.4s\n" /* outr00 = w3 * r1[0]*/ \
          "fmla   v16.4s ,  %[w3].4s,  v7.4s\n" /* outr01 = w3 * r1[1]*/ \
          "fmla   v17.4s ,  %[w3].4s,  v8.4s\n" /* outr02 = w3 * r1[2]*/ \
          "fmla   v18.4s ,  %[w3].4s,  v9.4s\n" /* outr03 = w3 * r1[3]*/ \
          "fmla   v19.4s ,  %[w3].4s,  v0.4s\n" /* outr10 = w3 * r2[0]*/ \
          "fmla   v20.4s ,  %[w3].4s,  v1.4s\n" /* outr11 = w3 * r2[1]*/ \
          "fmla   v21.4s ,  %[w3].4s,  v2.4s\n" /* outr12 = w3 * r2[2]*/ \
          "fmla   v22.4s ,  %[w3].4s,  v3.4s\n" /* outr13 = w3 * r2[3]*/ \
          /*  r1, r2, mul w4, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w4].4s,  v7.4s\n" /* outr00 = w4 * r1[1]*/ \
          "ldp    q6, q7,   [%[inr3]], #32\n"     /* load input r3*/ \
          "fmla   v16.4s ,  %[w4].4s,  v8.4s\n" /* outr01 = w4 * r1[2]*/ \
          "fmla   v17.4s ,  %[w4].4s,  v9.4s\n" /* outr02 = w4 * r1[3]*/ \
          "fmla   v18.4s ,  %[w4].4s,  v10.4s\n"/* outr03 = w4 * r1[4]*/ \
          "ldp    x0, x1, [%[outl]]  \n" \
          "fmla   v19.4s ,  %[w4].4s,  v1.4s\n" /* outr10 = w4 * r2[1]*/ \
          "fmla   v20.4s ,  %[w4].4s,  v2.4s\n" /* outr11 = w4 * r2[2]*/ \
          "fmla   v21.4s ,  %[w4].4s,  v3.4s\n" /* outr12 = w4 * r2[3]*/ \
          "fmla   v22.4s ,  %[w4].4s,  v4.4s\n" /* outr13 = w4 * r2[4]*/ \
          /*  r1, r2, mul w5, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w5].4s,  v8.4s\n" /* outr00 = w5 * r1[2]*/ \
          "fmla   v16.4s ,  %[w5].4s,  v9.4s\n" /* outr01 = w5 * r1[3]*/ \
          "ldp    q8, q9,   [%[inr3]], #32\n"     /* load input r3*/ \
          "fmla   v17.4s ,  %[w5].4s,  v10.4s\n"/* outr02 = w5 * r1[4]*/ \
          "fmla   v18.4s ,  %[w5].4s,  v11.4s\n"/* outr03 = w5 * r1[5]*/ \
          "ldp    q10, q11,   [%[inr3]]\n"        /* load input r3*/ \
          "fmla   v19.4s ,  %[w5].4s,  v2.4s\n" /* outr10 = w5 * r2[2]*/ \
          "fmla   v20.4s ,  %[w5].4s,  v3.4s\n" /* outr11 = w5 * r2[3]*/ \
          "fmla   v21.4s ,  %[w5].4s,  v4.4s\n" /* outr12 = w5 * r2[4]*/ \
          "fmla   v22.4s ,  %[w5].4s,  v5.4s\n" /* outr13 = w5 * r2[5]*/ \
          /*  r2, r3, mul w6, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w6].4s,  v0.4s\n" /* outr00 = w6 * r2[0]*/ \
          "fmla   v16.4s ,  %[w6].4s,  v1.4s\n" /* outr01 = w6 * r2[1]*/ \
          "fmla   v17.4s ,  %[w6].4s,  v2.4s\n" /* outr02 = w6 * r2[2]*/ \
          "fmla   v18.4s ,  %[w6].4s,  v3.4s\n" /* outr03 = w6 * r2[3]*/ \
          "ldp    x2, x3, [%[outl], #16]  \n" \
          "fmla   v19.4s ,  %[w6].4s,  v6.4s\n" /* outr10 = w6 * r3[0]*/ \
          "fmla   v20.4s ,  %[w6].4s,  v7.4s\n" /* outr11 = w6 * r3[1]*/ \
          "fmla   v21.4s ,  %[w6].4s,  v8.4s\n" /* outr12 = w6 * r3[2]*/ \
          "fmla   v22.4s ,  %[w6].4s,  v9.4s\n" /* outr13 = w6 * r3[3]*/ \
          /*  r2, r3, mul w7, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w7].4s,  v1.4s\n" /* outr00 = w7 * r2[1]*/ \
          "fmla   v16.4s ,  %[w7].4s,  v2.4s\n" /* outr01 = w7 * r2[2]*/ \
          "fmla   v17.4s ,  %[w7].4s,  v3.4s\n" /* outr02 = w7 * r2[3]*/ \
          "fmla   v18.4s ,  %[w7].4s,  v4.4s\n" /* outr03 = w7 * r2[4]*/ \
          "ldp    x4, x5, [%[outl], #32]  \n" \
          "fmla   v19.4s ,  %[w7].4s,  v7.4s\n" /* outr10 = w7 * r3[1]*/ \
          "fmla   v20.4s ,  %[w7].4s,  v8.4s\n" /* outr11 = w7 * r3[2]*/ \
          "fmla   v21.4s ,  %[w7].4s,  v9.4s\n" /* outr12 = w7 * r3[3]*/ \
          "fmla   v22.4s ,  %[w7].4s,  v10.4s\n"/* outr13 = w7 * r3[4]*/ \
          /*  r2, r3, mul w8, get out r0, r1 */ \
          "fmla   v15.4s ,  %[w8].4s,  v2.4s\n" /* outr00 = w8 * r2[2]*/ \
          "fmla   v16.4s ,  %[w8].4s,  v3.4s\n" /* outr01 = w8 * r2[3]*/ \
          "fmla   v17.4s ,  %[w8].4s,  v4.4s\n" /* outr02 = w8 * r2[0]*/ \
          "fmla   v18.4s ,  %[w8].4s,  v5.4s\n" /* outr03 = w8 * r2[1]*/ \
          "ldp    x6, x7, [%[outl], #48]  \n" \
          "fmla   v19.4s ,  %[w8].4s,  v8.4s\n" /* outr10 = w8 * r3[2]*/ \
          "fmla   v20.4s ,  %[w8].4s,  v9.4s\n" /* outr11 = w8 * r3[3]*/ \
          "fmla   v21.4s ,  %[w8].4s,  v10.4s\n"/* outr12 = w8 * r3[0]*/ \
          "fmla   v22.4s ,  %[w8].4s,  v11.4s\n"/* outr13 = w8 * r3[1]*/ \
          \
          "fadd   v15.4s, v15.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v16.4s, v16.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v17.4s, v17.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v18.4s, v18.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v19.4s, v19.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v20.4s, v20.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v21.4s, v21.4s, %[vbias].4s\n"/* add bias */ \
          "fadd   v22.4s, v22.4s, %[vbias].4s\n"/* add bias */ \
          /* transpose */ \
          "trn1   v0.4s, v15.4s, v16.4s\n" /* r0: a0a1c0c1*/ \
          "trn2   v1.4s, v15.4s, v16.4s\n" /* r0: b0b1d0d1*/ \
          "trn1   v2.4s, v17.4s, v18.4s\n" /* r0: a2a3c2c3*/ \
          "trn2   v3.4s, v17.4s, v18.4s\n" /* r0: b2b3d2d3*/ \
          "trn1   v4.4s, v19.4s, v20.4s\n" /* r1: a0a1c0c1*/ \
          "trn2   v5.4s, v19.4s, v20.4s\n" /* r1: b0b1d0d1*/ \
          "trn1   v6.4s, v21.4s, v22.4s\n" /* r1: a2a3c2c3*/ \
          "trn2   v7.4s, v21.4s, v22.4s\n" /* r1: b2b3d2d3*/ \
          "trn1   v15.2d, v0.2d, v2.2d\n"  /* r0: a0a1a2a3*/ \
          "trn2   v19.2d, v0.2d, v2.2d\n"  /* r0: c0c1c2c3*/ \
          "trn1   v17.2d, v1.2d, v3.2d\n"  /* r0: b0b1b2b3*/ \
          "trn2   v21.2d, v1.2d, v3.2d\n"  /* r0: d0d1d2d3*/ \
          "trn1   v16.2d, v4.2d, v6.2d\n"  /* r1: a0a1a2a3*/ \
          "trn2   v20.2d, v4.2d, v6.2d\n"  /* r1: c0c1c2c3*/ \
          "trn1   v18.2d, v5.2d, v7.2d\n"  /* r1: b0b1b2b3*/ \
          "trn2   v22.2d, v5.2d, v7.2d\n"  /* r1: d0d1d2d3*/

#define RELU \
          "movi   v0.4s, #0\n"             /* for relu */ \
          "ldr x0,    [%[outl], #80]\n" \
          "fmax   v15.4s, v15.4s, v0.4s\n" \
          "fmax   v16.4s, v16.4s, v0.4s\n" \
          "fmax   v17.4s, v17.4s, v0.4s\n" \
          "fmax   v18.4s, v18.4s, v0.4s\n" \
          "ld1 {v1.4s}, [x0]\n" \
          "fmax   v19.4s, v19.4s, v0.4s\n" \
          "fmax   v20.4s, v20.4s, v0.4s\n" \
          "fmax   v21.4s, v21.4s, v0.4s\n" \
          "fmax   v22.4s, v22.4s, v0.4s\n" \
          "ldr x0,    [%[outl]]\n" \

#define RELU6 \
          "fmin   v15.4s, v15.4s, v1.4s\n" \
          "fmin   v16.4s, v16.4s, v1.4s\n" \
          "fmin   v17.4s, v17.4s, v1.4s\n" \
          "fmin   v18.4s, v18.4s, v1.4s\n" \
          "fmin   v19.4s, v19.4s, v1.4s\n" \
          "fmin   v20.4s, v20.4s, v1.4s\n" \
          "fmin   v21.4s, v21.4s, v1.4s\n" \
          "fmin   v22.4s, v22.4s, v1.4s\n"

#define LEAKY_RELU \
          "movi   v0.4s, #0\n"             /* for relu */ \
          "ldr x0,    [%[outl], #88]\n" \
          "fcmge v1.4s, v15.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "fcmge v2.4s, v16.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "ld1 {v9.4s}, [x0] \n" \
          "fcmge v3.4s, v17.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "fcmge v4.4s, v18.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "ldr x0,    [%[outl]] \n" \
          "fmul v5.4s, v15.4s, v9.4s \n" /* mul */ \
          "fmul v6.4s, v16.4s, v9.4s \n" /* mul */ \
          "fmul v7.4s, v17.4s, v9.4s \n" /* mul */ \
          "fmul v8.4s, v18.4s, v9.4s \n" /* mul */ \
          "bif  v15.16b, v5.16b, v1.16b \n" /* choose*/ \
          "bif  v16.16b, v6.16b, v2.16b \n" /* choose*/ \
          "bif  v17.16b, v7.16b, v3.16b \n" /* choose*/ \
          "bif  v18.16b, v8.16b, v4.16b \n" /* choose*/ \
          "fcmge v1.4s, v19.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "fcmge v2.4s, v20.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "fcmge v3.4s, v21.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "fcmge v4.4s, v22.4s,  v0.4s \n" /* vcgeq_f32 */ \
          "fmul v5.4s, v19.4s, v9.4s \n" /* mul */ \
          "fmul v6.4s, v20.4s, v9.4s \n" /* mul */ \
          "fmul v7.4s, v21.4s, v9.4s \n" /* mul */ \
          "fmul v8.4s, v22.4s, v9.4s \n" /* mul */ \
          "bif  v19.16b, v5.16b, v1.16b \n" /* choose*/ \
          "bif  v20.16b, v6.16b, v2.16b \n" /* choose*/ \
          "bif  v21.16b, v7.16b, v3.16b \n" /* choose*/ \
          "bif  v22.16b, v8.16b, v4.16b \n" /* choose*/

#define STORE \
          "cbnz   %w[flag_mask], 1f\n" \
          "str    q15, [x0]\n" /* save outc00 */ \
          "str    q16, [x4]\n" /* save outc01 */ \
          "str    q17, [x1]\n" /* save outc10 */ \
          "str    q18, [x5]\n" /* save outc11 */ \
          "str    q19, [x2]\n" /* save outc20 */ \
          "str    q20, [x6]\n" /* save outc21 */ \
          "str    q21, [x3]\n" /* save outc30 */ \
          "str    q22, [x7]\n" /* save outc31 */ \
          "b 2f\n" \
          "1:\n" \
          "str  q15, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q17, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q19, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q21, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q16, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q18, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q20, [%[out]], #16 \n" /* save remain to pre_out */ \
          "str  q22, [%[out]], #16 \n" /* save remain to pre_out */ \
          "2:\n"
#else
#define COMPUTE \
          /* load weights */ \
          "vld1.32    {d10-d13}, [%[wc0]]!      @ load w0, w1, to q5, q6\n" \
          "vld1.32    {d14-d15}, [%[wc0]]!      @ load w2, to q7\n" \
          /* load r0, r1 */ \
          "vld1.32    {d0-d3}, [%[r0]]!         @ load r0, q0, q1\n" \
          "vld1.32    {d4-d7}, [%[r0]]!         @ load r0, q2, q3\n" \
          /* main loop */ \
          "0:                                   @ main loop\n" \
          /* mul r0 with w0, w1, w2, get out r0 */ \
          "vmul.f32   q8, q5, q0                @ w0 * inr00\n" \
          "vmul.f32   q9, q5, q1                @ w0 * inr01\n" \
          "vmul.f32   q10, q5, q2               @ w0 * inr02\n" \
          "vmul.f32   q11, q5, q3               @ w0 * inr03\n" \
          "vmla.f32   q8, q6, q1                @ w1 * inr01\n" \
          "vld1.32    {d0-d3}, [%[r0]]          @ load r0, q0, q1\n" \
          "vmla.f32   q9, q6, q2                @ w1 * inr02\n" \
          "vmla.f32   q10, q6, q3               @ w1 * inr03\n" \
          "vmla.f32   q11, q6, q0               @ w1 * inr04\n" \
          "vmla.f32   q8, q7, q2                @ w2 * inr02\n" \
          "vmla.f32   q9, q7, q3                @ w2 * inr03\n" \
          "vld1.32    {d4-d7}, [%[r1]]!         @ load r0, q2, q3\n" \
          "vmla.f32   q10, q7, q0               @ w2 * inr04\n" \
          "vmla.f32   q11, q7, q1               @ w2 * inr05\n" \
          "vld1.32    {d0-d3}, [%[r1]]!         @ load r0, q0, q1\n" \
          "vld1.32    {d8-d9}, [%[wc0]]!        @ load w3 to q4\n" \
          /* mul r1 with w0-w5, get out r0, r1 */ \
          "vmul.f32   q12, q5, q2               @ w0 * inr10\n" \
          "vmul.f32   q13, q5, q3               @ w0 * inr11\n" \
          "vmul.f32   q14, q5, q0               @ w0 * inr12\n" \
          "vmul.f32   q15, q5, q1               @ w0 * inr13\n" \
          "vld1.32    {d10-d11}, [%[wc0]]!      @ load w4 to q5\n" \
          "vmla.f32   q8, q4, q2                @ w3 * inr10\n" \
          "vmla.f32   q9, q4, q3                @ w3 * inr11\n" \
          "vmla.f32   q10, q4, q0               @ w3 * inr12\n" \
          "vmla.f32   q11, q4, q1               @ w3 * inr13\n" \
          /* mul r1 with w1, w4, get out r1, r0 */ \
          "vmla.f32   q8, q5, q3                @ w4 * inr11\n" \
          "vmla.f32   q12, q6, q3               @ w1 * inr11\n" \
          "vld1.32    {d4-d7}, [%[r1]]          @ load r1, q2, q3\n" \
          "vmla.f32   q9, q5, q0                @ w4 * inr12\n" \
          "vmla.f32   q13, q6, q0               @ w1 * inr12\n" \
          "vmla.f32   q10, q5, q1               @ w4 * inr13\n" \
          "vmla.f32   q14, q6, q1               @ w1 * inr13\n" \
          "vmla.f32   q11, q5, q2               @ w4 * inr14\n" \
          "vmla.f32   q15, q6, q2               @ w1 * inr14\n" \
          "vld1.32    {d12-d13}, [%[wc0]]!      @ load w5 to q6\n" \
          /* mul r1 with w2, w5, get out r1, r0 */ \
          "vmla.f32   q12, q7, q0               @ w2 * inr12\n" \
          "vmla.f32   q13, q7, q1               @ w2 * inr13\n" \
          "vmla.f32   q8, q6, q0                @ w5 * inr12\n" \
          "vmla.f32   q9, q6, q1                @ w5 * inr13\n" \
          "vld1.32    {d0-d3}, [%[r2]]!         @ load r2, q0, q1\n" \
          "vmla.f32   q14, q7, q2               @ w2 * inr14\n" \
          "vmla.f32   q15, q7, q3               @ w2 * inr15\n" \
          "vmla.f32   q10, q6, q2               @ w5 * inr14\n" \
          "vmla.f32   q11, q6, q3               @ w5 * inr15\n" \
          "vld1.32    {d4-d7}, [%[r2]]!         @ load r2, q0, q1\n" \
          "vld1.32    {d14-d15}, [%[wc0]]!      @ load w6, to q7\n" \
          /* mul r2 with w3-w8, get out r0, r1 */ \
          "vmla.f32   q12, q4, q0               @ w3 * inr20\n" \
          "vmla.f32   q13, q4, q1               @ w3 * inr21\n" \
          "vmla.f32   q14, q4, q2               @ w3 * inr22\n" \
          "vmla.f32   q15, q4, q3               @ w3 * inr23\n" \
          "vld1.32    {d8-d9}, [%[wc0]]!        @ load w7, to q4\n" \
          "vmla.f32   q8,  q7, q0               @ w6 * inr20\n" \
          "vmla.f32   q9,  q7, q1               @ w6 * inr21\n" \
          "vmla.f32   q10, q7, q2               @ w6 * inr22\n" \
          "vmla.f32   q11, q7, q3               @ w6 * inr23\n" \
          /* mul r2 with w4, w7, get out r1, r0 */ \
          "vmla.f32   q8,  q4, q1               @ w7 * inr21\n" \
          "vmla.f32   q12, q5, q1               @ w4 * inr21\n" \
          "vld1.32    {d0-d3}, [%[r2]]          @ load r2, q0, q1\n" \
          "vmla.f32   q9,  q4, q2               @ w7 * inr22\n" \
          "vmla.f32   q13, q5, q2               @ w4 * inr22\n" \
          "vmla.f32   q10, q4, q3               @ w7 * inr23\n" \
          "vmla.f32   q14, q5, q3               @ w4 * inr23\n" \
          "vmla.f32   q11, q4, q0               @ w7 * inr24\n" \
          "vmla.f32   q15, q5, q0               @ w4 * inr24\n" \
          "vld1.32    {d10-d11}, [%[wc0]]!      @ load w8 to q5\n" \
          /* mul r1 with w5, w8, get out r1, r0 */ \
          "vmla.f32   q12, q6, q2               @ w5 * inr22\n" \
          "vmla.f32   q13, q6, q3               @ w5 * inr23\n" \
          "vmla.f32   q8,  q5, q2               @ w8 * inr22\n" \
          "vmla.f32   q9,  q5, q3               @ w8 * inr23\n" \
          "vld1.32    {d4-d7}, [%[r3]]!         @ load r3, q2, q3\n" \
          "ldr r4,    [%[outl], #32]            @ load bias addr to r4\n" \
          "vmla.f32   q14, q6, q0               @ w5 * inr24\n" \
          "vmla.f32   q15, q6, q1               @ w5 * inr25\n" \
          "vmla.f32   q10, q5, q0               @ w8 * inr24\n" \
          "vmla.f32   q11, q5, q1               @ w8 * inr25\n" \
          "vld1.32    {d0-d3}, [%[r3]]!         @ load r3, q0, q1\n" \
          "sub %[wc0], %[wc0], #144      @ wc0 - 144 to start address\n" \
          /* mul r3 with w6, w7, w8, get out r1 */ \
          "vmla.f32   q12, q7, q2               @ w6 * inr30\n" \
          "vmla.f32   q13, q7, q3               @ w6 * inr31\n" \
          "vmla.f32   q14, q7, q0               @ w6 * inr32\n" \
          "vmla.f32   q15, q7, q1               @ w6 * inr33\n" \
          "vmla.f32   q12, q4, q3               @ w7 * inr31\n" \
          "vld1.32    {d4-d7}, [%[r3]]          @ load r3, q2, q3\n" \
          "vld1.32    {d12-d13}, [r4]           @ load bias\n" \
          "vmla.f32   q13, q4, q0               @ w7 * inr32\n" \
          "vmla.f32   q14, q4, q1               @ w7 * inr33\n" \
          "vmla.f32   q15, q4, q2               @ w7 * inr34\n" \
          "ldr r0,    [%[outl]]                 @ load outc00 to r0\n" \
          "vmla.f32   q12, q5, q0               @ w8 * inr32\n" \
          "vmla.f32   q13, q5, q1               @ w8 * inr33\n" \
          "vmla.f32   q14, q5, q2               @ w8 * inr34\n" \
          "vmla.f32   q15, q5, q3               @ w8 * inr35\n" \
          "ldr r1,    [%[outl], #4]             @ load outc10 to r1\n" \
          "vadd.f32   q8, q8, q6                @ r00 add bias\n" \
          "vadd.f32   q9, q9, q6                @ r01 add bias\n" \
          "vadd.f32   q10, q10, q6              @ r02 add bias\n" \
          "vadd.f32   q11, q11, q6              @ r03 add bias\n" \
          "ldr r2,    [%[outl], #8]             @ load outc20 to r2\n" \
          "vadd.f32   q12, q12, q6              @ r10 add bias\n" \
          "vadd.f32   q13, q13, q6              @ r11 add bias\n" \
          "vadd.f32   q14, q14, q6              @ r12 add bias\n" \
          "vadd.f32   q15, q15, q6              @ r13 add bias\n" \
          "ldr r3,    [%[outl], #12]            @ load outc30 to r3\n" \
          "vmov.u32   q7, #0                    @ mov zero to q7\n"
#define RELU \
          "vmax.f32  q8, q8, q7                 @ r00 relu\n" \
          "vmax.f32  q9, q9, q7                 @ r01 relu\n" \
          "vmax.f32  q10, q10, q7               @ r02 relu\n" \
          "vmax.f32  q11, q11, q7               @ r03 relu\n" \
          "vmax.f32  q12, q12, q7               @ r10 relu\n" \
          "vmax.f32  q13, q13, q7               @ r11 relu\n" \
          "vmax.f32  q14, q14, q7               @ r12 relu\n" \
          "vmax.f32  q15, q15, q7               @ r13 relu\n"

#define RELU6 \
          "ldr r4,    [%[outl], #40]            @ load six to r4\n" \
          "vld1.32 {d12-d13}, [r4] @load data \n" \
          "vmin.f32  q8, q8, q6                 @ r00 relu\n" \
          "vmin.f32  q9, q9, q6                 @ r01 relu\n" \
          "vmin.f32  q10, q10, q6               @ r02 relu\n" \
          "vmin.f32  q11, q11, q6               @ r03 relu\n" \
          "vmin.f32  q12, q12, q6               @ r10 relu\n" \
          "vmin.f32  q13, q13, q6               @ r11 relu\n" \
          "vmin.f32  q14, q14, q6               @ r12 relu\n" \
          "vmin.f32  q15, q15, q6               @ r13 relu\n"

#define LEAKY_RELU \
          "ldr r4,    [%[outl], #44]            @ load scale to r4\n" \
          "vld1.32 {d12-d13}, [r4] @load data \n" \
          "vcge.f32 q0, q8, q7        @ q0 > 0 \n"   \
          "vcge.f32 q1, q9, q7        @ q0 > 0 \n"   \
          "vmul.f32 q4, q8, q6 \n"  \
          "vmul.f32 q5, q9, q6 \n"  \
          "vcge.f32 q2, q10, q7        @ q0 > 0 \n"   \
          "vcge.f32 q3, q11, q7        @ q0 > 0 \n"   \
          "vbif q8, q4, q0 @ choose \n" \
          "vbif q9, q5, q1 @ choose \n" \
          "vmul.f32 q4, q10, q6 \n"  \
          "vmul.f32 q5, q11, q6 \n"  \
          "vbif q10, q4, q2 @ choose \n" \
          "vbif q11, q5, q3 @ choose \n" \
          "vcge.f32 q0, q12, q7        @ q0 > 0 \n"   \
          "vcge.f32 q1, q13, q7        @ q0 > 0 \n"   \
          "vmul.f32 q4, q12, q6 \n"  \
          "vmul.f32 q5, q13, q6 \n"  \
          "vcge.f32 q2, q14, q7        @ q0 > 0 \n"   \
          "vcge.f32 q3, q15, q7        @ q0 > 0 \n"   \
          "vbif q12, q4, q0 @ choose \n" \
          "vbif q13, q5, q1 @ choose \n" \
          "vmul.f32 q4, q14, q6 \n"  \
          "vmul.f32 q5, q15, q6 \n"  \
          "vbif q14, q4, q2 @ choose \n" \
          "vbif q15, q5, q3 @ choose \n"

#define STORE \
          "ldr r4,   [%[outl], #16]   @ load outc01 to r4\n" \
          "vtrn.32   q8, q9           @ r0: q8 : a0a1c0c1, q9 : b0b1d0d1\n" \
          "vtrn.32   q10, q11         @ r0: q10: a2a3c2c3, q11: b2b3d2d3\n" \
          "vtrn.32   q12, q13         @ r1: q12: a0a1c0c1, q13: b0b1d0d1\n" \
          "vtrn.32   q14, q15         @ r1: q14: a2a3c2c3, q15: b2b3d2d3\n" \
          "vswp      d17, d20         @ r0: q8 : a0a1a2a3, q10: c0c1c2c3 \n" \
          "vswp      d19, d22         @ r0: q9 : b0b1b2b3, q11: d0d1d2d3 \n" \
          "vswp      d25, d28         @ r1: q12: a0a1a2a3, q14: c0c1c2c3 \n" \
          "vswp      d27, d30         @ r1: q13: b0b1b2b3, q15: d0d1d2d3 \n" \
          "cmp %[flag_mask], #0       @ cmp flag mask\n" \
          "bne 2f\n" \
          "vst1.32   {d16-d17}, [r0]  @ save outc00\n" \
          "vst1.32   {d18-d19}, [r1]  @ save outc10\n" \
          "vst1.32   {d20-d21}, [r2]  @ save outc20\n" \
          "vst1.32   {d22-d23}, [r3]  @ save outc30\n" \
          "ldr r0,   [%[outl], #20]   @ load outc11 to r5\n" \
          "ldr r1,   [%[outl], #24]   @ load outc21 to r0\n" \
          "ldr r2,   [%[outl], #28]   @ load outc31 to r1\n" \
          "vst1.32   {d24-d25}, [r4]  @ save outc01\n" \
          "vst1.32   {d26-d27}, [r0]  @ save outc11\n" \
          "vst1.32   {d28-d29}, [r1]  @ save outc21\n" \
          "vst1.32   {d30-d31}, [r2]  @ save outc31\n" \
          "b 3f                       @ branch end\n" \
          "2: \n" \
          "vst1.32 {d16-d17}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d18-d19}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d20-d21}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d22-d23}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d24-d25}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d26-d27}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d28-d29}, [%[out0]]!  @ save remain to pre_out\n" \
          "vst1.32 {d30-d31}, [%[out0]]!  @ save remain to pre_out\n" \
          "3: \n"
#endif
// clang-format on
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
                               const operators::ActivationParam act_param,
                               ARMContext* ctx) {
  float six_ptr[4] = {0.f, 0.f, 0.f, 0.f};
  float scale_ptr[4] = {1.f, 1.f, 1.f, 1.f};
  float relu_ptr[4] = {0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    switch (act_param.active_type) {
      case lite_api::ActivationType::kRelu:
        conv_3x3s1_depthwise_fp32_relu(i_data,
                                       o_data,
                                       bs,
                                       oc,
                                       oh,
                                       ow,
                                       ic,
                                       ih,
                                       win,
                                       weights,
                                       bias,
                                       relu_ptr,
                                       six_ptr,
                                       scale_ptr,
                                       param,
                                       ctx);
        break;
      case lite_api::ActivationType::kRelu6:
        six_ptr[0] = act_param.Relu_clipped_coef;
        six_ptr[1] = act_param.Relu_clipped_coef;
        six_ptr[2] = act_param.Relu_clipped_coef;
        six_ptr[3] = act_param.Relu_clipped_coef;
        conv_3x3s1_depthwise_fp32_relu6(i_data,
                                        o_data,
                                        bs,
                                        oc,
                                        oh,
                                        ow,
                                        ic,
                                        ih,
                                        win,
                                        weights,
                                        bias,
                                        relu_ptr,
                                        six_ptr,
                                        scale_ptr,
                                        param,
                                        ctx);
        break;
      case lite_api::ActivationType::kLeakyRelu:
        scale_ptr[0] = act_param.Leaky_relu_alpha;
        scale_ptr[1] = act_param.Leaky_relu_alpha;
        scale_ptr[2] = act_param.Leaky_relu_alpha;
        scale_ptr[3] = act_param.Leaky_relu_alpha;
        conv_3x3s1_depthwise_fp32_leakyRelu(i_data,
                                            o_data,
                                            bs,
                                            oc,
                                            oh,
                                            ow,
                                            ic,
                                            ih,
                                            win,
                                            weights,
                                            bias,
                                            relu_ptr,
                                            six_ptr,
                                            scale_ptr,
                                            param,
                                            ctx);
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param.active_type)
                   << " fuse not support";
    }
  } else {
    conv_3x3s1_depthwise_fp32_bias(i_data,
                                   o_data,
                                   bs,
                                   oc,
                                   oh,
                                   ow,
                                   ic,
                                   ih,
                                   win,
                                   weights,
                                   bias,
                                   relu_ptr,
                                   six_ptr,
                                   scale_ptr,
                                   param,
                                   ctx);
  }
}

void conv_3x3s1_depthwise_fp32_bias(const float* i_data,
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
                                    float* relu_ptr,
                                    float* six_ptr,
                                    float* scale_ptr,
                                    const operators::ConvParam& param,
                                    ARMContext* ctx) {
  int threads = ctx->threads();

  auto paddings = *param.paddings;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];

  const int out_c_block = 4;
  const int out_h_kernel = 2;
  const int out_w_kernel = 4;
  const int win_ext = ow + 2;
  const int ow_round = ROUNDUP(ow, 4);
  const int win_round = ROUNDUP(win_ext, 4);
  const int hin_round = oh + 2;
  const int prein_size = win_round * hin_round * out_c_block;
  auto workspace_size =
      threads * prein_size + win_round /*tmp zero*/ + ow_round /*tmp writer*/;
  ctx->ExtendWorkspace(sizeof(float) * workspace_size);

  bool flag_bias = param.bias != nullptr;

  /// get workspace
  float* ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float* ptr_write = ptr_zero + win_round;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = ws + win_round;
  int hs = -pad_h;
  int he = hs + hin_round;
  int w_loop = ow_round / 4;
  auto remain = w_loop * 4 - ow;
  bool flag_remain = remain > 0;
  remain = 4 - remain;
  remain = remain > 0 ? remain : 0;
  int row_len = win_round * out_c_block;

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c += out_c_block) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      float pre_out[out_c_block * out_w_kernel * out_h_kernel];  // NOLINT
      prepack_input_nxwc4_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;
      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c + 1];
        bias_local[2] = bias[c + 2];
        bias_local[3] = bias[c + 3];
      }
      float32x4_t vbias = vld1q_f32(bias_local);
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 4);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 8);   // w2, v25
      float32x4_t w3 = vld1q_f32(weight_c + 12);  // w3, v26
      float32x4_t w4 = vld1q_f32(weight_c + 16);  // w4, v27
      float32x4_t w5 = vld1q_f32(weight_c + 20);  // w5, v28
      float32x4_t w6 = vld1q_f32(weight_c + 24);  // w6, v29
      float32x4_t w7 = vld1q_f32(weight_c + 28);  // w7, v30
      float32x4_t w8 = vld1q_f32(weight_c + 32);  // w8, v31
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc00 = dout_c00 + h * ow;
        float* outc01 = outc00 + ow;
        float* outc10 = outc00 + size_out_channel;
        float* outc11 = outc10 + ow;
        float* outc20 = outc10 + size_out_channel;
        float* outc21 = outc20 + ow;
        float* outc30 = outc20 + size_out_channel;
        float* outc31 = outc30 + ow;
        const float* inr0 = pre_din + h * row_len;
        const float* inr1 = inr0 + row_len;
        const float* inr2 = inr1 + row_len;
        const float* inr3 = inr2 + row_len;
        if (c + out_c_block > oc) {
          switch (c + out_c_block - oc) {
            case 3:  // outc10-outc30 is ptr_write and extra
              outc10 = ptr_write;
              outc11 = ptr_write;
            case 2:  // outc20-outc30 is ptr_write and extra
              outc20 = ptr_write;
              outc21 = ptr_write;
            case 1:  // outc30 is ptr_write and extra
              outc30 = ptr_write;
              outc31 = ptr_write;
            default:
              break;
          }
        }
        if (h + out_h_kernel > oh) {
          outc01 = ptr_write;
          outc11 = ptr_write;
          outc21 = ptr_write;
          outc31 = ptr_write;
        }

        float* outl[] = {outc00,
                         outc10,
                         outc20,
                         outc30,
                         outc01,
                         outc11,
                         outc21,
                         outc31,
                         reinterpret_cast<float*>(bias_local),
                         reinterpret_cast<float*>(relu_ptr),
                         reinterpret_cast<float*>(six_ptr),
                         reinterpret_cast<float*>(scale_ptr)};
        void* outl_ptr = reinterpret_cast<void*>(outl);
        for (int w = 0; w < w_loop; ++w) {
          bool flag_mask = (w == w_loop - 1) && flag_remain;
          float* out0 = pre_out;
#ifdef __aarch64__
          asm volatile(COMPUTE STORE
                       : [inr0] "+r"(inr0),
                         [inr1] "+r"(inr1),
                         [inr2] "+r"(inr2),
                         [inr3] "+r"(inr3),
                         [out] "+r"(out0)
                       : [w0] "w"(w0),
                         [w1] "w"(w1),
                         [w2] "w"(w2),
                         [w3] "w"(w3),
                         [w4] "w"(w4),
                         [w5] "w"(w5),
                         [w6] "w"(w6),
                         [w7] "w"(w7),
                         [w8] "w"(w8),
                         [vbias] "w"(vbias),
                         [outl] "r"(outl_ptr),
                         [flag_mask] "r"(flag_mask)
                       : "cc",
                         "memory",
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
                         "v11",
                         "v15",
                         "v16",
                         "v17",
                         "v18",
                         "v19",
                         "v20",
                         "v21",
                         "v22",
                         "x0",
                         "x1",
                         "x2",
                         "x3",
                         "x4",
                         "x5",
                         "x6",
                         "x7");
#else
          asm volatile(COMPUTE STORE
                       : [r0] "+r"(inr0),
                         [r1] "+r"(inr1),
                         [r2] "+r"(inr2),
                         [r3] "+r"(inr3),
                         [out0] "+r"(out0),
                         [wc0] "+r"(weight_c)
                       : [flag_mask] "r"(flag_mask), [outl] "r"(outl_ptr)
                       : "cc",
                         "memory",
                         "q0",
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
                         "r2",
                         "r3",
                         "r4");
#endif
          outl[0] += 4;
          outl[1] += 4;
          outl[2] += 4;
          outl[3] += 4;
          outl[4] += 4;
          outl[5] += 4;
          outl[6] += 4;
          outl[7] += 4;
          if (flag_mask) {
            memcpy(outl[0] - 4, pre_out, remain * sizeof(float));
            memcpy(outl[1] - 4, pre_out + 4, remain * sizeof(float));
            memcpy(outl[2] - 4, pre_out + 8, remain * sizeof(float));
            memcpy(outl[3] - 4, pre_out + 12, remain * sizeof(float));
            memcpy(outl[4] - 4, pre_out + 16, remain * sizeof(float));
            memcpy(outl[5] - 4, pre_out + 20, remain * sizeof(float));
            memcpy(outl[6] - 4, pre_out + 24, remain * sizeof(float));
            memcpy(outl[7] - 4, pre_out + 28, remain * sizeof(float));
          }
        }
      }
    }
  }
}

void conv_3x3s1_depthwise_fp32_relu(const float* i_data,
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
                                    float* relu_ptr,
                                    float* six_ptr,
                                    float* scale_ptr,
                                    const operators::ConvParam& param,
                                    ARMContext* ctx) {
  int threads = ctx->threads();

  auto paddings = *param.paddings;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];

  const int out_c_block = 4;
  const int out_h_kernel = 2;
  const int out_w_kernel = 4;
  const int win_ext = ow + 2;
  const int ow_round = ROUNDUP(ow, 4);
  const int win_round = ROUNDUP(win_ext, 4);
  const int hin_round = oh + 2;
  const int prein_size = win_round * hin_round * out_c_block;
  auto workspace_size =
      threads * prein_size + win_round /*tmp zero*/ + ow_round /*tmp writer*/;
  ctx->ExtendWorkspace(sizeof(float) * workspace_size);

  bool flag_bias = param.bias != nullptr;

  /// get workspace
  float* ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float* ptr_write = ptr_zero + win_round;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = ws + win_round;
  int hs = -pad_h;
  int he = hs + hin_round;
  int w_loop = ow_round / 4;
  auto remain = w_loop * 4 - ow;
  bool flag_remain = remain > 0;
  remain = 4 - remain;
  remain = remain > 0 ? remain : 0;
  int row_len = win_round * out_c_block;

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c += out_c_block) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      float pre_out[out_c_block * out_w_kernel * out_h_kernel];  // NOLINT
      prepack_input_nxwc4_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;
      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c + 1];
        bias_local[2] = bias[c + 2];
        bias_local[3] = bias[c + 3];
      }
      float32x4_t vbias = vld1q_f32(bias_local);
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 4);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 8);   // w2, v25
      float32x4_t w3 = vld1q_f32(weight_c + 12);  // w3, v26
      float32x4_t w4 = vld1q_f32(weight_c + 16);  // w4, v27
      float32x4_t w5 = vld1q_f32(weight_c + 20);  // w5, v28
      float32x4_t w6 = vld1q_f32(weight_c + 24);  // w6, v29
      float32x4_t w7 = vld1q_f32(weight_c + 28);  // w7, v30
      float32x4_t w8 = vld1q_f32(weight_c + 32);  // w8, v31
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc00 = dout_c00 + h * ow;
        float* outc01 = outc00 + ow;
        float* outc10 = outc00 + size_out_channel;
        float* outc11 = outc10 + ow;
        float* outc20 = outc10 + size_out_channel;
        float* outc21 = outc20 + ow;
        float* outc30 = outc20 + size_out_channel;
        float* outc31 = outc30 + ow;
        const float* inr0 = pre_din + h * row_len;
        const float* inr1 = inr0 + row_len;
        const float* inr2 = inr1 + row_len;
        const float* inr3 = inr2 + row_len;
        if (c + out_c_block > oc) {
          switch (c + out_c_block - oc) {
            case 3:  // outc10-outc30 is ptr_write and extra
              outc10 = ptr_write;
              outc11 = ptr_write;
            case 2:  // outc20-outc30 is ptr_write and extra
              outc20 = ptr_write;
              outc21 = ptr_write;
            case 1:  // outc30 is ptr_write and extra
              outc30 = ptr_write;
              outc31 = ptr_write;
            default:
              break;
          }
        }
        if (h + out_h_kernel > oh) {
          outc01 = ptr_write;
          outc11 = ptr_write;
          outc21 = ptr_write;
          outc31 = ptr_write;
        }

        float* outl[] = {outc00,
                         outc10,
                         outc20,
                         outc30,
                         outc01,
                         outc11,
                         outc21,
                         outc31,
                         reinterpret_cast<float*>(bias_local),
                         reinterpret_cast<float*>(relu_ptr),
                         reinterpret_cast<float*>(six_ptr),
                         reinterpret_cast<float*>(scale_ptr)};
        void* outl_ptr = reinterpret_cast<void*>(outl);
        for (int w = 0; w < w_loop; ++w) {
          bool flag_mask = (w == w_loop - 1) && flag_remain;
          float* out0 = pre_out;
#ifdef __aarch64__
          asm volatile(COMPUTE RELU STORE
                       : [inr0] "+r"(inr0),
                         [inr1] "+r"(inr1),
                         [inr2] "+r"(inr2),
                         [inr3] "+r"(inr3),
                         [out] "+r"(out0)
                       : [w0] "w"(w0),
                         [w1] "w"(w1),
                         [w2] "w"(w2),
                         [w3] "w"(w3),
                         [w4] "w"(w4),
                         [w5] "w"(w5),
                         [w6] "w"(w6),
                         [w7] "w"(w7),
                         [w8] "w"(w8),
                         [vbias] "w"(vbias),
                         [outl] "r"(outl_ptr),
                         [flag_mask] "r"(flag_mask)
                       : "cc",
                         "memory",
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
                         "v11",
                         "v15",
                         "v16",
                         "v17",
                         "v18",
                         "v19",
                         "v20",
                         "v21",
                         "v22",
                         "x0",
                         "x1",
                         "x2",
                         "x3",
                         "x4",
                         "x5",
                         "x6",
                         "x7");
#else
          asm volatile(COMPUTE RELU STORE
                       : [r0] "+r"(inr0),
                         [r1] "+r"(inr1),
                         [r2] "+r"(inr2),
                         [r3] "+r"(inr3),
                         [out0] "+r"(out0),
                         [wc0] "+r"(weight_c)
                       : [flag_mask] "r"(flag_mask), [outl] "r"(outl_ptr)
                       : "cc",
                         "memory",
                         "q0",
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
                         "r2",
                         "r3",
                         "r4");
#endif
          outl[0] += 4;
          outl[1] += 4;
          outl[2] += 4;
          outl[3] += 4;
          outl[4] += 4;
          outl[5] += 4;
          outl[6] += 4;
          outl[7] += 4;
          if (flag_mask) {
            memcpy(outl[0] - 4, pre_out, remain * sizeof(float));
            memcpy(outl[1] - 4, pre_out + 4, remain * sizeof(float));
            memcpy(outl[2] - 4, pre_out + 8, remain * sizeof(float));
            memcpy(outl[3] - 4, pre_out + 12, remain * sizeof(float));
            memcpy(outl[4] - 4, pre_out + 16, remain * sizeof(float));
            memcpy(outl[5] - 4, pre_out + 20, remain * sizeof(float));
            memcpy(outl[6] - 4, pre_out + 24, remain * sizeof(float));
            memcpy(outl[7] - 4, pre_out + 28, remain * sizeof(float));
          }
        }
      }
    }
  }
}

void conv_3x3s1_depthwise_fp32_relu6(const float* i_data,
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
                                     float* relu_ptr,
                                     float* six_ptr,
                                     float* scale_ptr,
                                     const operators::ConvParam& param,
                                     ARMContext* ctx) {
  int threads = ctx->threads();

  auto paddings = *param.paddings;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];

  const int out_c_block = 4;
  const int out_h_kernel = 2;
  const int out_w_kernel = 4;
  const int win_ext = ow + 2;
  const int ow_round = ROUNDUP(ow, 4);
  const int win_round = ROUNDUP(win_ext, 4);
  const int hin_round = oh + 2;
  const int prein_size = win_round * hin_round * out_c_block;
  auto workspace_size =
      threads * prein_size + win_round /*tmp zero*/ + ow_round /*tmp writer*/;
  ctx->ExtendWorkspace(sizeof(float) * workspace_size);

  bool flag_bias = param.bias != nullptr;

  /// get workspace
  float* ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float* ptr_write = ptr_zero + win_round;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = ws + win_round;
  int hs = -pad_h;
  int he = hs + hin_round;
  int w_loop = ow_round / 4;
  auto remain = w_loop * 4 - ow;
  bool flag_remain = remain > 0;
  remain = 4 - remain;
  remain = remain > 0 ? remain : 0;
  int row_len = win_round * out_c_block;

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c += out_c_block) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      float pre_out[out_c_block * out_w_kernel * out_h_kernel];  // NOLINT
      prepack_input_nxwc4_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;
      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c + 1];
        bias_local[2] = bias[c + 2];
        bias_local[3] = bias[c + 3];
      }
      float32x4_t vbias = vld1q_f32(bias_local);
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 4);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 8);   // w2, v25
      float32x4_t w3 = vld1q_f32(weight_c + 12);  // w3, v26
      float32x4_t w4 = vld1q_f32(weight_c + 16);  // w4, v27
      float32x4_t w5 = vld1q_f32(weight_c + 20);  // w5, v28
      float32x4_t w6 = vld1q_f32(weight_c + 24);  // w6, v29
      float32x4_t w7 = vld1q_f32(weight_c + 28);  // w7, v30
      float32x4_t w8 = vld1q_f32(weight_c + 32);  // w8, v31
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc00 = dout_c00 + h * ow;
        float* outc01 = outc00 + ow;
        float* outc10 = outc00 + size_out_channel;
        float* outc11 = outc10 + ow;
        float* outc20 = outc10 + size_out_channel;
        float* outc21 = outc20 + ow;
        float* outc30 = outc20 + size_out_channel;
        float* outc31 = outc30 + ow;
        const float* inr0 = pre_din + h * row_len;
        const float* inr1 = inr0 + row_len;
        const float* inr2 = inr1 + row_len;
        const float* inr3 = inr2 + row_len;
        if (c + out_c_block > oc) {
          switch (c + out_c_block - oc) {
            case 3:  // outc10-outc30 is ptr_write and extra
              outc10 = ptr_write;
              outc11 = ptr_write;
            case 2:  // outc20-outc30 is ptr_write and extra
              outc20 = ptr_write;
              outc21 = ptr_write;
            case 1:  // outc30 is ptr_write and extra
              outc30 = ptr_write;
              outc31 = ptr_write;
            default:
              break;
          }
        }
        if (h + out_h_kernel > oh) {
          outc01 = ptr_write;
          outc11 = ptr_write;
          outc21 = ptr_write;
          outc31 = ptr_write;
        }

        float* outl[] = {outc00,
                         outc10,
                         outc20,
                         outc30,
                         outc01,
                         outc11,
                         outc21,
                         outc31,
                         reinterpret_cast<float*>(bias_local),
                         reinterpret_cast<float*>(relu_ptr),
                         reinterpret_cast<float*>(six_ptr),
                         reinterpret_cast<float*>(scale_ptr)};
        void* outl_ptr = reinterpret_cast<void*>(outl);
        for (int w = 0; w < w_loop; ++w) {
          bool flag_mask = (w == w_loop - 1) && flag_remain;
          float* out0 = pre_out;
#ifdef __aarch64__
          asm volatile(COMPUTE RELU RELU6 STORE
                       : [inr0] "+r"(inr0),
                         [inr1] "+r"(inr1),
                         [inr2] "+r"(inr2),
                         [inr3] "+r"(inr3),
                         [out] "+r"(out0)
                       : [w0] "w"(w0),
                         [w1] "w"(w1),
                         [w2] "w"(w2),
                         [w3] "w"(w3),
                         [w4] "w"(w4),
                         [w5] "w"(w5),
                         [w6] "w"(w6),
                         [w7] "w"(w7),
                         [w8] "w"(w8),
                         [vbias] "w"(vbias),
                         [outl] "r"(outl_ptr),
                         [flag_mask] "r"(flag_mask)
                       : "cc",
                         "memory",
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
                         "v11",
                         "v15",
                         "v16",
                         "v17",
                         "v18",
                         "v19",
                         "v20",
                         "v21",
                         "v22",
                         "x0",
                         "x1",
                         "x2",
                         "x3",
                         "x4",
                         "x5",
                         "x6",
                         "x7");
#else
          asm volatile(COMPUTE RELU RELU6 STORE
                       : [r0] "+r"(inr0),
                         [r1] "+r"(inr1),
                         [r2] "+r"(inr2),
                         [r3] "+r"(inr3),
                         [out0] "+r"(out0),
                         [wc0] "+r"(weight_c)
                       : [flag_mask] "r"(flag_mask), [outl] "r"(outl_ptr)
                       : "cc",
                         "memory",
                         "q0",
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
                         "r2",
                         "r3",
                         "r4");
#endif
          outl[0] += 4;
          outl[1] += 4;
          outl[2] += 4;
          outl[3] += 4;
          outl[4] += 4;
          outl[5] += 4;
          outl[6] += 4;
          outl[7] += 4;
          if (flag_mask) {
            memcpy(outl[0] - 4, pre_out, remain * sizeof(float));
            memcpy(outl[1] - 4, pre_out + 4, remain * sizeof(float));
            memcpy(outl[2] - 4, pre_out + 8, remain * sizeof(float));
            memcpy(outl[3] - 4, pre_out + 12, remain * sizeof(float));
            memcpy(outl[4] - 4, pre_out + 16, remain * sizeof(float));
            memcpy(outl[5] - 4, pre_out + 20, remain * sizeof(float));
            memcpy(outl[6] - 4, pre_out + 24, remain * sizeof(float));
            memcpy(outl[7] - 4, pre_out + 28, remain * sizeof(float));
          }
        }
      }
    }
  }
}

void conv_3x3s1_depthwise_fp32_leakyRelu(const float* i_data,
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
                                         float* relu_ptr,
                                         float* six_ptr,
                                         float* scale_ptr,
                                         const operators::ConvParam& param,
                                         ARMContext* ctx) {
  int threads = ctx->threads();

  auto paddings = *param.paddings;
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];

  const int out_c_block = 4;
  const int out_h_kernel = 2;
  const int out_w_kernel = 4;
  const int win_ext = ow + 2;
  const int ow_round = ROUNDUP(ow, 4);
  const int win_round = ROUNDUP(win_ext, 4);
  const int hin_round = oh + 2;
  const int prein_size = win_round * hin_round * out_c_block;
  auto workspace_size =
      threads * prein_size + win_round /*tmp zero*/ + ow_round /*tmp writer*/;
  ctx->ExtendWorkspace(sizeof(float) * workspace_size);

  bool flag_bias = param.bias != nullptr;

  /// get workspace
  float* ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float* ptr_write = ptr_zero + win_round;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = ws + win_round;
  int hs = -pad_h;
  int he = hs + hin_round;
  int w_loop = ow_round / 4;
  auto remain = w_loop * 4 - ow;
  bool flag_remain = remain > 0;
  remain = 4 - remain;
  remain = remain > 0 ? remain : 0;
  int row_len = win_round * out_c_block;

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c += out_c_block) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      float pre_out[out_c_block * out_w_kernel * out_h_kernel];  // NOLINT
      prepack_input_nxwc4_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;
      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c + 1];
        bias_local[2] = bias[c + 2];
        bias_local[3] = bias[c + 3];
      }
      float32x4_t vbias = vld1q_f32(bias_local);
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 4);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 8);   // w2, v25
      float32x4_t w3 = vld1q_f32(weight_c + 12);  // w3, v26
      float32x4_t w4 = vld1q_f32(weight_c + 16);  // w4, v27
      float32x4_t w5 = vld1q_f32(weight_c + 20);  // w5, v28
      float32x4_t w6 = vld1q_f32(weight_c + 24);  // w6, v29
      float32x4_t w7 = vld1q_f32(weight_c + 28);  // w7, v30
      float32x4_t w8 = vld1q_f32(weight_c + 32);  // w8, v31
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc00 = dout_c00 + h * ow;
        float* outc01 = outc00 + ow;
        float* outc10 = outc00 + size_out_channel;
        float* outc11 = outc10 + ow;
        float* outc20 = outc10 + size_out_channel;
        float* outc21 = outc20 + ow;
        float* outc30 = outc20 + size_out_channel;
        float* outc31 = outc30 + ow;
        const float* inr0 = pre_din + h * row_len;
        const float* inr1 = inr0 + row_len;
        const float* inr2 = inr1 + row_len;
        const float* inr3 = inr2 + row_len;
        if (c + out_c_block > oc) {
          switch (c + out_c_block - oc) {
            case 3:  // outc10-outc30 is ptr_write and extra
              outc10 = ptr_write;
              outc11 = ptr_write;
            case 2:  // outc20-outc30 is ptr_write and extra
              outc20 = ptr_write;
              outc21 = ptr_write;
            case 1:  // outc30 is ptr_write and extra
              outc30 = ptr_write;
              outc31 = ptr_write;
            default:
              break;
          }
        }
        if (h + out_h_kernel > oh) {
          outc01 = ptr_write;
          outc11 = ptr_write;
          outc21 = ptr_write;
          outc31 = ptr_write;
        }

        float* outl[] = {outc00,
                         outc10,
                         outc20,
                         outc30,
                         outc01,
                         outc11,
                         outc21,
                         outc31,
                         reinterpret_cast<float*>(bias_local),
                         reinterpret_cast<float*>(relu_ptr),
                         reinterpret_cast<float*>(six_ptr),
                         reinterpret_cast<float*>(scale_ptr)};
        void* outl_ptr = reinterpret_cast<void*>(outl);
        for (int w = 0; w < w_loop; ++w) {
          bool flag_mask = (w == w_loop - 1) && flag_remain;
          float* out0 = pre_out;
#ifdef __aarch64__
          asm volatile(COMPUTE LEAKY_RELU STORE
                       : [inr0] "+r"(inr0),
                         [inr1] "+r"(inr1),
                         [inr2] "+r"(inr2),
                         [inr3] "+r"(inr3),
                         [out] "+r"(out0)
                       : [w0] "w"(w0),
                         [w1] "w"(w1),
                         [w2] "w"(w2),
                         [w3] "w"(w3),
                         [w4] "w"(w4),
                         [w5] "w"(w5),
                         [w6] "w"(w6),
                         [w7] "w"(w7),
                         [w8] "w"(w8),
                         [vbias] "w"(vbias),
                         [outl] "r"(outl_ptr),
                         [flag_mask] "r"(flag_mask)
                       : "cc",
                         "memory",
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
                         "v11",
                         "v15",
                         "v16",
                         "v17",
                         "v18",
                         "v19",
                         "v20",
                         "v21",
                         "v22",
                         "x0",
                         "x1",
                         "x2",
                         "x3",
                         "x4",
                         "x5",
                         "x6",
                         "x7");
#else
          asm volatile(COMPUTE LEAKY_RELU STORE
                       : [r0] "+r"(inr0),
                         [r1] "+r"(inr1),
                         [r2] "+r"(inr2),
                         [r3] "+r"(inr3),
                         [out0] "+r"(out0),
                         [wc0] "+r"(weight_c)
                       : [flag_mask] "r"(flag_mask), [outl] "r"(outl_ptr)
                       : "cc",
                         "memory",
                         "q0",
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
                         "r2",
                         "r3",
                         "r4");
#endif
          outl[0] += 4;
          outl[1] += 4;
          outl[2] += 4;
          outl[3] += 4;
          outl[4] += 4;
          outl[5] += 4;
          outl[6] += 4;
          outl[7] += 4;
          if (flag_mask) {
            memcpy(outl[0] - 4, pre_out, remain * sizeof(float));
            memcpy(outl[1] - 4, pre_out + 4, remain * sizeof(float));
            memcpy(outl[2] - 4, pre_out + 8, remain * sizeof(float));
            memcpy(outl[3] - 4, pre_out + 12, remain * sizeof(float));
            memcpy(outl[4] - 4, pre_out + 16, remain * sizeof(float));
            memcpy(outl[5] - 4, pre_out + 20, remain * sizeof(float));
            memcpy(outl[6] - 4, pre_out + 24, remain * sizeof(float));
            memcpy(outl[7] - 4, pre_out + 28, remain * sizeof(float));
          }
        }
      }
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
