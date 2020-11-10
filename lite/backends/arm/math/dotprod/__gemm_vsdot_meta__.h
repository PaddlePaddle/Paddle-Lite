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

#pragma once

// clang-format off
#define GEMM_VSDOT_INT8_KERNEL                                           \
  "vld1.s8  {q0}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "vld1.s8  {d2}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "veor.s32    q4,  q4, q4     \n"     /* out0 = 0 */                    \
  "veor.s32    q5,  q5, q5     \n"     /* out0 = 0 */                    \
  "veor.s32    q6,  q6, q6     \n"     /* out0 = 0 */                    \
  "veor.s32    q7,  q7, q7     \n"     /* out0 = 0 */                    \
  "veor.s32    q8,  q8, q8     \n"     /* out0 = 0 */                    \
  "veor.s32    q9,  q9, q9     \n"     /* out0 = 0 */                    \
  "veor.s32    q10,  q10, q10  \n"     /* out0 = 0 */                    \
  "veor.s32    q11,  q11, q11  \n"     /* out0 = 0 */                    \
  "veor.s32    q12,  q12, q12  \n"     /* out0 = 0 */                    \
  "veor.s32    q13,  q13, q13  \n"     /* out0 = 0 */                    \
  "veor.s32    q14,  q14, q14  \n"     /* out0 = 0 */                    \
  "veor.s32    q15,  q15, q15  \n"     /* out0 = 0 */                    \
  "cmp   %[k], #0              \n"                                       \
  "beq   2f                    \n"                                       \
  "1:                          \n"                                       \
  "vld1.s8  {q2}, [%[b_ptr]]!  \n"                                       \
  "vld1.s8  {q3}, [%[b_ptr]]!  \n"                                       \
  "vsdot.s8 q4,   q2,  d0[0]   \n"                                       \
  "vsdot.s8 q6,   q2,  d0[1]   \n"                                       \
  "vsdot.s8 q8,   q2,  d1[0]   \n"                                       \
  "vsdot.s8 q10,  q2,  d1[1]   \n"                                       \
  "vsdot.s8 q12,  q2,  d2[0]   \n"                                       \
  "vsdot.s8 q14,  q2,  d2[1]   \n"                                       \
  "vsdot.s8 q5,   q3,  d0[0]   \n"                                       \
  "vsdot.s8 q7,   q3,  d0[1]   \n"                                       \
  "vsdot.s8 q9,   q3,  d1[0]   \n"                                       \
  "vsdot.s8 q11,  q3,  d1[1]   \n"                                       \
  "vsdot.s8 q13,  q3,  d2[0]   \n"                                       \
  "vsdot.s8 q15,  q3,  d2[1]   \n"                                       \
  "vld1.s8  {q0}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "vld1.s8  {d2}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "subs %[k], %[k], #1         \n"                                       \
  "bne    1b                   \n"                                       \
  "2:                          \n"                                    

  
// #define GEMM_DOT_CVT_INT32_TO_FP32                                       \
//   "vld1.32  {d0-d1}, [%[scale]]!    \n"                                  \
//   "vld1.32  {d2-d3}, [%[bias_ptr]]! \n"                                  \
//   "vcvt.f32.s32     q2, q4          \n"                                  \
//   "vcvt.f32.s32     q3, q5          \n"                                  \
//   "vdup.32    q4,   d2[0]           \n"                                  \
//   "vdup.32    q5,   d2[0]           \n"                                  \
//   "vmla.f32   q4,   q2, d0[0]       \n"                                  \
//   "vmla.f32   q5,   q3, d0[0]       \n"                                  \
//   "vcvt.f32.s32     q2, q6          \n"                                  \
//   "vcvt.f32.s32     q3, q7          \n"                                  \
//   "vdup.32    q6,   d2[1]           \n"                                  \
//   "vdup.32    q7,   d2[1]           \n"                                  \
//   "vmla.f32   q6,   q2, d0[1]       \n"                                  \
//   "vmla.f32   q7,   q3, d0[1]       \n"                                  \
//   "vcvt.f32.s32     q2, q8          \n"                                  \
//   "vcvt.f32.s32     q3, q9          \n"                                  \
//   "vdup.32    q8,   d3[0]           \n"                                  \
//   "vdup.32    q9,   d3[0]           \n"                                  \
//   "vmla.f32   q8,   q2, d1[0]       \n"                                  \
//   "vmla.f32   q9,   q3, d1[0]       \n"                                  \
//   "vcvt.f32.s32     q2, q10         \n"                                  \
//   "vcvt.f32.s32     q3, q11         \n"                                  \
//   "vdup.32    q10,  d3[1]           \n"                                  \
//   "vdup.32    q11,  d3[1]           \n"                                  \
//   "vmla.f32   q10,  q2, d1[1]       \n"                                  \
//   "vmla.f32   q11,  q3, d1[1]       \n"                                  \
//   "vld1.32  {d0}, [%[scale]]        \n"                                  \
//   "vld1.32  {d2}, [%[bias_ptr]]     \n"                                  \
//   "vcvt.f32.s32     q2, q12         \n"                                  \
//   "vcvt.f32.s32     q3, q13         \n"                                  \
//   "vdup.32    q12,  d2[0]           \n"                                  \
//   "vdup.32    q13,  d2[0]           \n"                                  \
//   "vmla.f32   q12,  q2, d0[0]       \n"                                  \
//   "vmla.f32   q13,  q3, d0[0]       \n"                                  \
//   "vcvt.f32.s32     q2, q14         \n"                                  \
//   "vcvt.f32.s32     q3, q15         \n"                                  \
//   "vdup.32    q14,  d2[1]           \n"                                  \
//   "vdup.32    q15,  d2[1]           \n"                                  \
//   "vmla.f32   q14,  q2, d0[1]       \n"                                  \
//   "vmla.f32   q15,  q3, d0[1]       \n"                                  

//   #define GEMM_DOT_ST_FP32                                               \
//   "vst1.I32 {q4}, [%[c_ptr0]]! \n"                                       \
//   "vst1.I32 {q6}, [%[c_ptr1]]! \n"                                       \
//   "vst1.I32 {q5}, [%[c_ptr0]]! \n"                                       \
//   "vst1.I32 {q7}, [%[c_ptr1]]! \n"                                       \
//   "vst1.I32 {q8}, [%[c_ptr2]]! \n"                                       \
//   "vst1.I32 {q9}, [%[c_ptr2]]! \n"                                       \
//   "vst1.I32 {q10},[%[c_ptr3]]! \n"                                       \
//   "vst1.I32 {q11},[%[c_ptr3]]! \n"                                       \
//   "vst1.I32 {q12},[%[c_ptr4]]! \n"                                       \
//   "vst1.I32 {q13},[%[c_ptr4]]! \n"                                       \
//   "vst1.I32 {q14},[%[c_ptr5]]! \n"                                       \
//   "vst1.I32 {q15},[%[c_ptr5]]! \n"                


// #define GEMM_DOT_RELU                                 \
//   "cmp    %[relu],   #0      \n"     /* skip relu */  \
//   "beq    12f                \n"                      \
//   "cmp    %[relu],    #1     \n"     /* skip relu */  \
//   "bne    13f                \n"     /* other act */  \
//   "vmov.f32   q0, #0.0       \n"     /* for relu*/    \
//   "vmax.f32   q4,   q4,   q0 \n"     /* relu*/        \
//   "vmax.f32   q5,   q5,   q0 \n"     /* relu*/        \
//   "vmax.f32   q6,   q6,   q0 \n"     /* relu*/        \
//   "vmax.f32   q7,   q7,   q0 \n"     /* relu*/        \
//   "vmax.f32   q8,   q8,   q0 \n"     /* relu*/        \
//   "vmax.f32   q9,   q9,   q0 \n"     /* relu*/        \
//   "vmax.f32   q10,  q10,  q0 \n"     /* relu*/        \
//   "vmax.f32   q11,  q11,  q0 \n"     /* relu*/        \
//   "vmax.f32   q12,  q12,  q0 \n"     /* relu*/        \
//   "vmax.f32   q13,  q13,  q0 \n"     /* relu*/        \
//   "vmax.f32   q14,  q14,  q0 \n"     /* relu*/        \
//   "vmax.f32   q15,  q15,  q0 \n"     /* relu*/        \
//   "b      12f                \n"     /* relu end */

// #define GEMM_DOT_RELU6                          \
//   "13:                       \n"                \
//   "cmp    %[relu],   #2\n"     /* skip relu6 */ \
//   "bne   14f\n"                                 \
//   "vmov.f32   q0, #0.0\n"       /* for relu*/   \
//   "vmax.f32   q4,   q4,   q0 \n"   /* relu*/    \
//   "vmax.f32   q5,   q5,   q0 \n"   /* relu*/    \
//   "vmax.f32   q6,   q6,   q0 \n"   /* relu*/    \
//   "vmax.f32   q7,   q7,   q0 \n"   /* relu*/    \
//   "vld1.32    {d2-d3}, [%[alpha]]! \n"          \
//   "vmax.f32   q8,   q8,   q0 \n"   /* relu*/    \
//   "vmax.f32   q9,   q9,   q0 \n"   /* relu*/    \
//   "vmax.f32   q10,  q10,  q0 \n"   /* relu*/    \
//   "vmax.f32   q11,  q11,  q0 \n"   /* relu*/    \
//   "vmax.f32   q12,  q12,  q0 \n"   /* relu*/    \
//   "vmax.f32   q13,  q13,  q0 \n"   /* relu*/    \
//   "vmax.f32   q14,  q14,  q0 \n"   /* relu*/    \
//   "vmax.f32   q15,  q15,  q0 \n"   /* relu*/    \
//   "vmin.f32   q4,   q4,   q1 \n"   /* relu6*/   \
//   "vmin.f32   q5,   q5,   q1 \n"   /* relu6*/   \
//   "vmin.f32   q6,   q6,   q1 \n"   /* relu6*/   \
//   "vmin.f32   q7,   q7,   q1 \n"   /* relu6*/   \
//   "vmin.f32   q8,   q8,   q1 \n"   /* relu6*/   \
//   "vmin.f32   q9,   q9,   q1 \n"   /* relu6*/   \
//   "vmin.f32   q10,  q10,  q1 \n"   /* relu6*/   \
//   "vmin.f32   q11,  q11,  q1 \n"   /* relu6*/   \
//   "vmin.f32   q12,  q12,  q1 \n"   /* relu6*/   \
//   "vmin.f32   q13,  q13,  q1 \n"   /* relu6*/   \
//   "vmin.f32   q14,  q14,  q1 \n"   /* relu6*/   \
//   "vmin.f32   q15,  q15,  q1 \n"   /* relu6*/   \
//   "b      12f                \n"   /* relu6 end */

// #define GEMM_DOT_LEAKY_RELU                      \
//   "14:                      \n"                  \
//   "vmov.f32   q0, #0.0      \n"       /* for leakyrelu*/   \
//   "vld1.32  {d2-d3}, [%[alpha]]! \n" /* leakyrelu alpha */ \
//   "vcge.f32 q2,   q4,   q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q4,   q1  \n" /* vmulq_f32 */  \
//   "vbif     q4,   q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q5,   q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q5,   q1  \n" /* vmulq_f32 */  \
//   "vbif     q5,   q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q6,   q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q6,   q1  \n" /* vmulq_f32 */  \
//   "vbif     q6,   q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q7,   q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q7,   q1  \n" /* vmulq_f32 */  \
//   "vbif     q7,   q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q8,   q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q8,   q1  \n" /* vmulq_f32 */  \
//   "vbif     q8,   q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q9,   q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q9,   q1  \n" /* vmulq_f32 */  \
//   "vbif     q9,   q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q10,  q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q10,  q1  \n" /* vmulq_f32 */  \
//   "vbif     q10,  q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q11,  q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q11,  q1  \n" /* vmulq_f32 */  \
//   "vbif     q11,  q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q12,  q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q12,  q1  \n" /* vmulq_f32 */  \
//   "vbif     q12,  q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q13,  q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q13,  q1  \n" /* vmulq_f32 */  \
//   "vbif     q13,  q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q14,  q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q14,  q1  \n" /* vmulq_f32 */  \
//   "vbif     q14,  q3,   q2  \n" /* choose*/      \
//   "vcge.f32 q2,   q15,  q0  \n" /* vcgeq_f32 */  \
//   "vmla.f32 q3,   q15,  q1  \n" /* vmulq_f32 */  \
//   "vbif     q15,  q3,   q2  \n" /* choose*/      \
//   "12:                      \n"


// #define GEMM_DOT_ST_INT8                                        \
//   "add %[alpha],    #16             \n"                         \
//   "vld1.32    {d0-d1},    [%[alpha]]\n"                         \
//   "vmov.f32   q1,   #0.5            \n"                         \
//   "vmov.f32   q2,   #-0.5           \n"                         \
//   "vcgt.f32   q3,   q4,   #0        \n"                         \
//   "vbif.f32   q1,   q2,   q3        \n"                         \
//   "vadd.f32   q4,   q1,   q4        \n"                         \
//   "vmov.f32   q1,   #0.5            \n"                         \
//   "vcgt.f32   q3,   q5,   #0        \n"                         \
//   "vbif.f32   q1,   q2,   q3        \n"                         \
//   "vadd.f32   q5,   q1,   q5        \n"                         \
//   /* data >= -127 */                                            \
//   "vcge.f32   q1,   q4,   q0        \n"                         \
//   "vcge.f32   q2,   q5,   q0        \n"                         \
//   "vbif q4,   q0,   q1              \n"                         \
//   "vbif q5,   q0,   q2              \n"                         \
//   /* fp32 to int32 */                                           \
//   "vcvt.s32.f32     q1,   q4        \n"                         \
//   "vcvt.s32.f32     q2,   q5        \n"                         \
//   /* int32 to int16 */                                          \
//   "vqmovn.s32 d8,   q1              \n"                         \
//   "vqmovn.s32 d9,   q2              \n"                         \
//   /* int16 to int8 */                                           \
//   "vqmovn.s16 d2,   q4              \n"                         \
//   "vst1.32    {d2}, [%[c_ptr0]]!    \n"                         \
//                                                                 \
//   "vmov.f32   q1,   #0.5            \n"                         \
//   "vmov.f32   q3,   #0.5            \n"                         \
//   "vmov.f32   q2,   #-0.5           \n"                         \
//   "vcgt.f32   q4,   q6,   #0        \n"                         \
//   "vcgt.f32   q5,   q7,   #0        \n"                         \
//   "vbif.f32   q1,   q2,   q4        \n"                         \
//   "vbif.f32   q3,   q2,   q5        \n"                         \
//   "vmov.f32   q4,   #0.5            \n"                         \
//   "vmov.f32   q5,   #0.5            \n"                         \
//   "vadd.f32   q6,   q1,   q6        \n"                         \
//   "vadd.f32   q7,   q3,   q7        \n"                         \
//   "vcgt.f32   q5,   q8,   #0        \n"                         \
//   "vbif.f32   q4,   q2,   q5        \n"                         \
//   "vadd.f32   q8,   q4,   q8        \n"                         \
//   "vcgt.f32   q5,   q9,   #0        \n"                         \
//   /* data >= -127 */                                            \
//   "vcge.f32   q1,   q6,   q0        \n"                         \
//   "vcge.f32   q3,   q7,   q0        \n"                         \
//   "vcge.f32   q4,   q8,   q0        \n"                         \
//   "vbif q6,   q0,   q1              \n"                         \
//   "vbif q7,   q0,   q3              \n"                         \
//   "vbif q8,   q0,   q4              \n"                         \
//   /* fp32 to int32 */                                           \
//   "vcvt.s32.f32     q1,   q6        \n"                         \
//   "vcvt.s32.f32     q3,   q7        \n"                         \
//   "vcvt.s32.f32     q4,   q8        \n"                         \
//   /* int32 to int16 */                                          \
//   "vqmovn.s32 d12,  q1              \n"                         \
//   "vqmovn.s32 d13,  q3              \n"                         \
//   "vqmovn.s32 d16,  q4              \n"                         \
//   "vmov.f32   q7,   #0.5            \n"                         \
//   "vbif.f32   q7,   q2,   q5        \n"                         \
//   "vadd.f32   q9,   q7,   q9        \n"                         \
//   "vcge.f32   q5,   q9,   q0        \n"                         \
//   "vbif q9,   q0,   q5              \n"                         \
//   "vcvt.s32.f32     q5,   q9        \n"                         \
//   "vqmovn.s32 d17,  q5              \n"                         \
//   /* int16 to int8 */                                           \
//   "vqmovn.s16 d19,  q8              \n"                         \
//   "vqmovn.s16 d18,  q6              \n"                         \
//   "vst1.32    {d18},[%[c_ptr1]]!    \n"                         \
//   "vst1.32    {d19},[%[c_ptr2]]!    \n"                         \
//                                                                 \
//   "vmov.f32   q2,   #-0.5           \n"                         \
//   "vmov.f32   q1,   #0.5            \n"                         \
//   "vmov.f32   q3,   #0.5            \n"                         \
//   "vmov.f32   q4,   #0.5            \n"                         \
//   "vmov.f32   q5,   #0.5            \n"                         \
//   "vcgt.f32   q6,   q10,  #0        \n"                         \
//   "vcgt.f32   q7,   q11,  #0        \n"                         \
//   "vcgt.f32   q8,   q12,  #0        \n"                         \
//   "vcgt.f32   q9,   q13,  #0        \n"                         \
//   "vbif.f32   q1,   q2,   q6        \n"                         \
//   "vbif.f32   q3,   q2,   q7        \n"                         \
//   "vbif.f32   q4,   q2,   q8        \n"                         \
//   "vbif.f32   q5,   q2,   q9        \n"                         \
//   "vmov.f32   q6,   #0.5            \n"                         \
//   "vmov.f32   q7,   #0.5            \n"                         \
//   "vcgt.f32   q8,   q14,  #0        \n"                         \
//   "vcgt.f32   q9,   q15,  #0        \n"                         \
//   "vbif.f32   q6,   q2,   q8        \n"                         \
//   "vbif.f32   q7,   q2,   q9        \n"                         \
//   "vadd.f32   q10,  q1,   q10       \n"                         \
//   "vadd.f32   q11,  q3,   q11       \n"                         \
//   "vadd.f32   q12,  q4,   q12       \n"                         \
//   "vadd.f32   q13,  q5,   q13       \n"                         \
//   "vadd.f32   q14,  q6,   q14       \n"                         \
//   "vadd.f32   q15,  q7,   q15       \n"                         \
//                                                                 \
//   "vcge.f32   q1,   q10,  q0        \n"                         \
//   "vcge.f32   q3,   q11,  q0        \n"                         \
//   "vcge.f32   q4,   q12,  q0        \n"                         \
//   "vcge.f32   q5,   q13,  q0        \n"                         \
//   "vcge.f32   q6,   q14,  q0        \n"                         \
//   "vcge.f32   q7,   q15,  q0        \n"                         \
//   "vbif       q10,  q0,   q1        \n"                         \
//   "vbif       q11,  q0,   q3        \n"                         \
//   "vbif       q12,  q0,   q4        \n"                         \
//   "vbif       q13,  q0,   q5        \n"                         \
//   "vbif       q14,  q0,   q6        \n"                         \
//   "vbif       q15,  q0,   q7        \n"                         \
//   /* fp32 to int32 */                                           \
//   "vcvt.s32.f32     q1,   q10       \n"                         \
//   "vcvt.s32.f32     q3,   q11       \n"                         \
//   "vcvt.s32.f32     q4,   q12       \n"                         \
//   "vcvt.s32.f32     q5,   q13       \n"                         \
//   "vcvt.s32.f32     q6,   q14       \n"                         \
//   "vcvt.s32.f32     q7,   q15       \n"                         \
//   /* int32 to int16 */                                          \
//   "vqmovn.s32 d16,  q1              \n"                         \
//   "vqmovn.s32 d17,  q3              \n"                         \
//   "vqmovn.s32 d18,  q4              \n"                         \
//   "vqmovn.s32 d19,  q5              \n"                         \
//   "vqmovn.s32 d20,  q6              \n"                         \
//   "vqmovn.s32 d21,  q7              \n"                         \
//   /* int16 to int8 */                                           \
//   "vqmovn.s16 d2,   q8              \n"                         \
//   "vqmovn.s16 d3,   q9              \n"                         \
//   "vqmovn.s16 d4,   q10             \n"                         \
//   "sub %[alpha], #16                \n"                         \
//   "vst1.32    {d2}, [%[c_ptr3]]!    \n"                         \
//   "vst1.32    {d3}, [%[c_ptr4]]!    \n"                         \
//   "vst1.32    {d4}, [%[c_ptr5]]!    \n"                                           

