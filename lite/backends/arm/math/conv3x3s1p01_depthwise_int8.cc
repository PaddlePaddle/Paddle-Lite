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
#include "lite/backends/arm/math/conv_depthwise.h"
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

// clang-format off
#ifdef __aarch64__
#define INT8_INIT_S1                                            \
  "PRFM PLDL1KEEP, [%[din_ptr0]]\n"                            \
  "PRFM PLDL1KEEP, [%[din_ptr1]]\n"                            \
  "PRFM PLDL1KEEP, [%[din_ptr2]]\n"                            \
  "PRFM PLDL1KEEP, [%[din_ptr3]]\n"                            \
  "ld1 {v0.8b}, [%[din_ptr0]], #8\n"                           \
  "ld1 {v2.8b}, [%[din_ptr1]], #8\n"                           \
  "ld1 {v4.8b}, [%[din_ptr2]], #8\n"                           \
  "ld1 {v6.8b}, [%[din_ptr3]], #8\n"                           \
  "dup v12.4s,  %[bias_val]\n"                                 \
  "dup v13.4s,  %[bias_val]\n"                                 \
  "dup v14.4s,  %[bias_val]\n"                                 \
  "dup v15.4s,  %[bias_val]\n"                                 \
  "ld1 {v1.8b}, [%[din_ptr0]]\n"                               \
  "ld1 {v3.8b}, [%[din_ptr1]]\n"                               \
  "ld1 {v5.8b}, [%[din_ptr2]]\n"                               \
  "ld1 {v7.8b}, [%[din_ptr3]]\n"                               \
  "movi v23.4s, #0\n"

#define INT8_LEFT_COMPUTE_S1                                    \
  "smull v16.8h, %[v1].8b, v0.8b\n"     /*a=r0_01234 * w[0][1]*/\
  "smull v17.8h, %[v1].8b, v2.8b\n"     /*b=r1_01234 * w[0][1]*/\
  "smull v18.8h, %[v7].8b, v4.8b\n"     /*c=r2_01234 * w[2][1]*/\
  "smull v19.8h, %[v7].8b, v8.8b\n"     /*d=r3_01234 * w[2][1]*/\
  "ext v8.8b, v23.8b, v0.8b, #7\n"                              \
  "ext v9.8b, v23.8b, v2.8b, #7\n"                              \
  "ext v10.8b, v23.8b, v4.8b, #7\n"                             \
  "ext v11.8b, v23.8b, v6.8b, #7\n"                             \
  "smlal v16.8h, %[v4].8b, v2.8b\n"     /*a+=r1_01234*w[1][1]*/ \
  "smlal v17.8h, %[v4].8b, v4.8b\n"     /*b+=r2_01234*w[1][1]*/ \
  "smlal v18.8h, %[v0].8b, v8.8b\n"     /*c+=r0_00123*w[0][0]*/ \
  "smlal v19.8h, %[v0].8b, v9.8b\n"     /*d+=r1_00123*w[0][0]*/ \
  "saddw v20.4s, v20.4s, v16.4h\n"                              \
  "saddw v21.4s, v21.4s, v17.4h\n"                              \
  "smull v22.8h, %[v6].8b, v10.8b\n"    /*e=r2_00123*w[2][0]*/  \
  "smull v24.8h, %[v6].8b, v11.8b\n"    /*f=r3_00123*w[2][0]*/  \
  "ext v8.8b, v0.8b, v1.8b, #1\n"                               \
  "ext v11.8b, v6.8b, v7.8b, #1\n"                              \
  "saddw2 v25.4s, v25.4s, v16.8h\n"      /*out0 += a*/          \
  "saddw2 v26.4s, v26.4s, v17.8h\n"      /*out1 += b*/          \
  "smull v16.8h, %[v3].8b, v9.8b\n"      /*a=r1_00123*w[1][0]*/ \
  "smull v17.8h, %[v3].8b, v10.8b\n"     /*b=r2_00123*w[1][0]*/ \
  "ext v9.8b, v2.8b, v3.8b, #1\n"                               \
  "ext v10.8b, v4.8b, v5.8b, #1\n"                              \
  "saddw v20.4s, v20.4s, v18.4h\n"                              \
  "saddw v21.4s, v21.4s, v19.4h\n"                              \
  "smlal v22.8h, %[v2].8b, v8.8b\n"      /*e+=r0_1234*w[0][2]*/\
  "smlal v24.8h, %[v2].8b, v9.8b\n"      /*f+=r1_1234*w[0][2]*/\
  "smlal v16.8h, %[v8].8b, v10.8b\n"     /*a+=r2_1234*w[2][2]*/\
  "smlal v17.8h, %[v8].8b, v11.8b\n"     /*b+=r3_1234*w[2][2]*/\
  "saddw2 v25.4s, v25.4s, v18.8h\n"      /*out0 += c*/          \
  "saddw2 v26.4s, v26.4s, v19.8h\n"      /*out1 += d*/          \
  "smull v18.8h, %[v5].8b, v3.8b\n"      /*c=r1_1234 * w[1][2]*/\
  "smull v19.8h, %[v5].8b, v5.8b\n"      /*d=r2_1234 * w[1][2]*/\
  "sub   %[din_ptr0], %[din_ptr0], #1\n"                        \
  "sub   %[din_ptr1], %[din_ptr1], #1\n"                        \
  "saddw v20.4s, v20.4s, v22.4h\n"                              \
  "saddw v21.4s, v21.4s, v24.4h\n"                              \
  "sub   %[din_ptr2], %[din_ptr2], #1\n"                        \
  "sub   %[din_ptr3], %[din_ptr3], #1\n"                        \
  "saddw2  v25.4s, v25.4s, v22.8h\n"                            \
  "saddw2  v26.4s, v26.4s, v24.8h\n"                            \
  "ld1 {v0.8b}, [%[din_ptr0]], #8\n"                            \
  "ld1 {v2.8b}, [%[din_ptr1]], #8\n"                            \
  "saddw v20.4s, v20.4s, v16.4h\n"       /*out0 += e*/          \
  "saddw v21.4s, v21.4s, v17.4h\n"       /*out1 += f*/          \
  "ld1 {v4.8b}, [%[din_ptr2]], #8\n"                            \
  "ld1 {v6.8b}, [%[din_ptr3]], #8\n"                            \
  "saddw2 v25.4s, v25.4s, v16.8h\n"       /*out1 += a*/          \
  "saddw2 v26.4s, v26.4s, v17.8h\n"       /*out1 += b*/          \
  "ld1 {v1.8b}, [%[din_ptr0]]\n"                                \
  "ld1 {v3.8b}, [%[din_ptr1]]\n"                                \
  "saddw v20.4s, v20.4s, v18.4h\n"                              \
  "saddw v21.4s, v21.4s, v19.4h\n"                              \
  "ld1 {v5.8b}, [%[din_ptr2]]\n"                                \
  "ld1 {v7.8b}, [%[din_ptr3]]\n"                                \
  "saddw2 v25.4s, v25.4s, v18.8h\n"      /*out1 += c*/          \
  "saddw2 v26.4s, v26.4s, v19.8h\n"      /*out1 += d*/

#define TRANS_INT32_TO_FP32_S1                                  \
  "ldr    q16, [%[scale]]\n"             /* load scale */       \
  "scvtf  v10.4s , v20.4s\n"             /*convert to fp32 */   \
  "scvtf  v11.4s , v21.4s\n"             /*convert to fp32 */   \
  "scvtf  v18.4s , v25.4s\n"             /*convert to fp32 */   \
  "scvtf  v19.4s , v26.4s\n"             /*convert to fp32 */   \
  "cmp  %w[is_relu],    #0\n"          /* skip relu */          \
  "fmlal v12.4s, v10.4s, v16.4s\n"                              \
  "fmlal v13.4s, v11.4s, v16.4s\n"                              \
  "fmlal v14.4s, v18.4s, v16.4s\n"                              \
  "fmlal v15.4s, v19.4s, v16.4s\n"

#define S1_RELU                                                 \
  "beq 0f\n"                           /* no act*/              \
  "cmp  %w[is_relu],    #1\n"          /* skip relu */          \
  "bne 1f\n"                           /* no relu*/             \
  "fmax v12.4s, v12.4s, v23.4s\n"                               \
  "fmax v13.4s, v13.4s, v23.4s\n"                               \
  "fmax v14.4s, v14.4s, v23.4s\n"                               \
  "fmax v15.4s, v15.4s, v23.4s\n"                               \
  "b    0f\n"
   
#define S1_RELU6                                                \
  "1:   \n"                                                     \
  "cmp  %w[is_relu],    #2\n"          /* skip relu */          \
  "ld1    {v8.4s}, [%[alpha]]\n"       /* relu6 alpha */        \
  "bne  2f\n"                                                   \
  "fmax v12.4s, v12.4s, v23.4s\n"                               \
  "fmax v13.4s, v13.4s, v23.4s\n"                               \
  "fmax v14.4s, v14.4s, v23.4s\n"                               \
  "fmax v15.4s, v15.4s, v23.4s\n"                               \
  "fmin v12.4s, v12.4s, v8.4s\n"                               \
  "fmin v13.4s, v13.4s, v8.4s\n"                               \
  "fmin v14.4s, v14.4s, v8.4s\n"                               \
  "fmin v15.4s, v15.4s, v8.4s\n"                               \
  "b    0f\n"

#define S1_LEAKY_RELU                                           \
  "2:    \n"                                                    \
  "fcmge v9.4s, v12.4s, v23.4s\n"      /* vcgeq_f32 */          \
  "fcmge v10.4s, v13.4s, v23.4s\n"     /* vcgeq_f32 */          \
  "fcmge v16.4s, v14.4s, v23.4s\n"      /* vcgeq_f32 */          \
  "fcmge v17.4s, v15.4s, v23.4s\n"     /* vcgeq_f32 */          \
  "fmul  v18.4s, v12.4s, v8.4s\n"      /* mul */                \
  "fmul  v19.4s, v13.4s, v8.4s\n"      /* mul */                \
  "fmul  v22.4s, v14.4s, v8.4s\n"      /* mul */                \
  "fmul  v24.4s, v15.4s, v8.4s\n"      /* mul */                \
  "bif   v12.16b, v18.16b, v9.16b\n"                            \
  "bif   v13.16b, v19.16b, v10.16b\n"                           \
  "bif   v14.16b, v22.16b, v9.16b\n"                            \
  "bif   v15.16b, v24.16b, v10.16b\n"                           \
  "0:    \n"

#define FP32_TO_INT8_S1                                         \
  "ld1 {v8.4s}, [%[vmax]]\n"                                    \
  /* data >= -127 */                                            \
  "fcmge v16.4s, v12.4s, v8.4s\n"                               \
  "fcmge v17.4s, v13.4s, v8.4s\n"                               \
  "fcmge v18.4s, v14.4s, v8.4s\n"                               \
  "fcmge v19.4s, v15.4s, v8.4s\n"                               \
  "bif v12.16b, v8.16b, v16.16b\n"                              \
  "bif v13.16b, v8.16b, v17.16b\n"                              \
  "bif v14.16b, v8.16b, v18.16b\n"                              \
  "bif v15.16b, v8.16b, v19.16b\n"                              \
  "fcvtas v16.4s, v12.4s\n"                   /*fp32 to int32*/ \
  "fcvtas v17.4s, v13.4s\n"                   /*fp32 to int32*/ \
  "fcvtas v18.4s, v14.4s\n"                   /*fp32 to int32*/ \
  "fcvtas v19.4s, v15.4s\n"                   /*fp32 to int32*/ \
  "sqxtn  v12.4h, v16.4s\n"                   /*int32 to int16*/\
  "sqxtn  v13.4h, v17.4s\n"                   /*int32 to int16*/\
  "sqxtn2  v12.8h, v18.4s\n"                  /*int32 to int16*/\
  "sqxtn2  v13.8h, v19.4s\n"                  /*int32 to int16*/\
  "sqxtn  v16.8b, v12.8h\n"                   /*int16 to int8 */\
  "sqxtn  v17.8b, v13.8h\n"                   /*int16 to int8 */

#define LEFT_RESULT_INT8_FP32_OUT_S1                            \
  INT8_LEFT_COMPUTE_S1                                          \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  "stp q12, q14, [%[ptr_out0]], #32\n"                          \
  "stp q13, q15, [%[ptr_out1]], #32\n"                          \
  "cmp  %w[cnt], #1\n"                                          \
  "dup v12.4s,  %[bias_val]\n"                                 \
  "dup v13.4s,  %[bias_val]\n"                                 \
  "dup v14.4s,  %[bias_val]\n"                                 \
  "dup v15.4s,  %[bias_val]\n"                                 \
  "blt 4f\n"

#define LEFT_RESULT_INT8_INT8_OUT_S1                            \
  INT8_LEFT_COMPUTE_S1                                          \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  FP32_TO_INT8_S1                                               \
  "st1 {v16.8b}, [%[ptr_out0]], #8\n"                           \
  "st1 {v17.8b}, [%[ptr_out1]], #8\n"                           \
  "cmp  %w[cnt], #1\n"                                          \
  "dup v12.4s,  %[bias_val]\n"                                 \
  "dup v13.4s,  %[bias_val]\n"                                 \
  "dup v14.4s,  %[bias_val]\n"                                 \
  "dup v15.4s,  %[bias_val]\n"                                 \
  "blt 4f\n"                                                    \

#define INT8_MID_COMPUTE_S1                                     \
  "3:\n"                                                        \
  "smull v16.8h, %[v0].8b, v0.8b\n"     /*a=r0_01234 * w[0][0]*/\
  "smull v17.8h, %[v0].8b, v2.8b\n"     /*b=r1_01234 * w[0][0]*/\
  "smull v18.8h, %[v6].8b, v4.8b\n"     /*c=r2_01234 * w[2][0]*/\
  "smull v19.8h, %[v6].8b, v8.8b\n"     /*d=r3_01234 * w[2][0]*/\
  "ext v8.8b, v0.8b, v1.8b, #1\n"                               \
  "ext v9.8b, v2.8b, v3.8b, #1\n"                               \
  "ext v10.8b, v4.8b, v5.8b, #1\n"                              \
  "ext v11.8b, v6.8b, v7.8b, #1\n"                              \
  "smlal v16.8h, %[v3].8b, v2.8b\n"     /*a+=r1_01234*w[1][0]*/ \
  "smlal v17.8h, %[v3].8b, v4.8b\n"     /*b+=r2_01234*w[1][0]*/ \
  "smlal v18.8h, %[v1].8b, v8.8b\n"     /*c+=r0_1234*w[0][1]*/  \
  "smlal v19.8h, %[v1].8b, v9.8b\n"     /*d+=r1_1234*w[0][1]*/  \
  "saddw v20.4s, v20.4s, v16.4h\n"                              \
  "saddw v21.4s, v21.4s, v17.4h\n"                              \
  "smull v22.8h, %[v7].8b, v10.8b\n"    /*e=r2_1234*w[2][1]*/   \
  "smull v24.8h, %[v7].8b, v11.8b\n"    /*f=r3_1234*w[2][1]*/   \
  "ext v8.8b, v0.8b, v1.8b, #2\n"                               \
  "ext v11.8b, v6.8b, v7.8b, #2\n"                              \
  "saddw2 v25.4s, v25.4s, v16.8h\n"      /*out0 += a*/          \
  "saddw2 v26.4s, v26.4s, v17.8h\n"      /*out1 += b*/          \
  "smull v16.8h, %[v4].8b, v9.8b\n"      /*a=r1_00123*w[1][1]*/ \
  "smull v17.8h, %[v4].8b, v10.8b\n"     /*b=r2_00123*w[1][1]*/ \
  "ext v9.8b, v2.8b, v3.8b, #2\n"                               \
  "ext v10.8b, v4.8b, v5.8b, #2\n"                              \
  "saddw v20.4s, v20.4s, v18.4h\n"                              \
  "saddw v21.4s, v21.4s, v19.4h\n"                              \
  "smlal v22.8h, %[v2].8b, v8.8b\n"      /*e+=r0_2345*w[0][2]*/\
  "smlal v24.8h, %[v2].8b, v9.8b\n"      /*f+=r1_2345*w[0][2]*/\
  "smlal v16.8h, %[v8].8b, v10.8b\n"     /*a+=r2_2345*w[2][2]*/\
  "smlal v17.8h, %[v8].8b, v11.8b\n"     /*b+=r3_2345*w[2][2]*/\
  "saddw2 v25.4s, v25.4s, v18.8h\n"      /*out0 += c*/          \
  "saddw2 v26.4s, v26.4s, v19.8h\n"      /*out1 += d*/          \
  "smull v18.8h, %[v5].8b, v3.8b\n"      /*c=r1_2345 * w[1][2]*/\
  "smull v19.8h, %[v5].8b, v5.8b\n"      /*d=r2_2345 * w[1][2]*/\
  "saddw v20.4s, v20.4s, v22.4h\n"                              \
  "saddw v21.4s, v21.4s, v24.4h\n"                              \
  "saddw2 v25.4s, v25.4s, v22.8h\n"                             \
  "saddw2 v26.4s, v26.4s, v24.8h\n"                             \
  "ld1 {v0.8b}, [%[din_ptr0]], #8\n"                            \
  "ld1 {v2.8b}, [%[din_ptr1]], #8\n"                            \
  "saddw v20.4s, v20.4s, v16.4h\n"       /*out0 += e*/          \
  "saddw v21.4s, v21.4s, v17.4h\n"       /*out1 += f*/          \
  "ld1 {v4.8b}, [%[din_ptr2]], #8\n"                            \
  "ld1 {v6.8b}, [%[din_ptr3]], #8\n"                            \
  "saddw2 v25.4s, v25.4s, v16.8h\n"       /*out1 += a*/         \
  "saddw2 v26.4s, v26.4s, v17.8h\n"       /*out1 += b*/         \
  "ld1 {v1.8b}, [%[din_ptr0]]\n"                                \
  "ld1 {v3.8b}, [%[din_ptr1]]\n"                                \
  "saddw v20.4s, v20.4s, v18.4h\n"                              \
  "saddw v21.4s, v21.4s, v19.4h\n"                              \
  "ld1 {v5.8b}, [%[din_ptr2]]\n"                                \
  "ld1 {v7.8b}, [%[din_ptr3]]\n"                                \
  "saddw2 v25.4s, v25.4s, v18.8h\n"      /*out1 += c*/          \
  "saddw2 v26.4s, v26.4s, v19.8h\n"      /*out1 += d*/

#define MID_RESULT_INT8_FP32_OUT_S1                             \
  INT8_MID_COMPUTE_S1                                           \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  "stp q12, q14, [%[ptr_out0]], #32\n"                          \
  "stp q13, q15, [%[ptr_out1]], #32\n"                          \
  "subs %w[cnt], %w[cnt], #1\n"                                 \
  "dup v12.4s,  %[bias_val]\n"                                 \
  "dup v13.4s,  %[bias_val]\n"                                 \
  "dup v14.4s,  %[bias_val]\n"                                 \
  "dup v15.4s,  %[bias_val]\n"                                 \
  "bne 3b\n"

#define MID_RESULT_INT8_INT8_OUT_S1                             \
  INT8_MID_COMPUTE_S1                                           \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  FP32_TO_INT8_S1                                               \
  "st1 {v16.8b}, [%[ptr_out0]], #8\n"                           \
  "st1 {v17.8b}, [%[ptr_out1]], #8\n"                           \
  "subs %w[cnt], %w[cnt], #1\n"                                 \
  "dup v12.4s,  %[bias_val]\n"                                 \
  "dup v13.4s,  %[bias_val]\n"                                 \
  "dup v14.4s,  %[bias_val]\n"                                 \
  "dup v15.4s,  %[bias_val]\n"                                 \
  "bne 3b\n"

#define INT8_RIGHT_COMPUTE_S1                                   \
  "4:  \n"                                                      \
  "ld1 {v16.8b}, [%[vmask]], #8\n"                              \
  "ld1 {v17.8b}, [%[vmask]]\n"                                  \
  "ld1 {v27.4s}, [%[rmask], #16\n"                              \
  "ld1 {v28.4s}, [%[rmask]\n"                                   \
  "bif v0.8b, v23.8b, v16.8b\n"                                 \
  "bif v2.8b, v23.8b, v16.8b\n"                                 \
  "bif v4.8b, v23.8b, v16.8b\n"                                 \
  "bif v6.8b, v23.8b, v16.8b\n"                                 \
  "smull v16.8h, %[v0].8b, v0.8b\n"     /*a=r0_01234 * w[0][0]*/\
  "smull v17.8h, %[v0].8b, v2.8b\n"     /*b=r1_01234 * w[0][0]*/\
  "smull v18.8h, %[v6].8b, v4.8b\n"     /*c=r2_01234 * w[2][0]*/\
  "smull v19.8h, %[v6].8b, v8.8b\n"     /*d=r3_01234 * w[2][0]*/\
  "bif v1.8b, v23.8b, v17.8b\n"                                 \
  "bif v3.8b, v23.8b, v17.8b\n"                                 \
  "bif v5.8b, v23.8b, v17.8b\n"                                 \
  "bif v7.8b, v23.8b, v17.8b\n"                                 \
  "smlal v16.8h, %[v3].8b, v2.8b\n"     /*a+=r1_01234*w[1][0]*/ \
  "ext v8.8b, v0.8b, v1.8b, #1\n"                               \
  "ext v9.8b, v2.8b, v3.8b, #1\n"                               \
  "ext v10.8b, v4.8b, v5.8b, #1\n"                              \
  "ext v11.8b, v6.8b, v7.8b, #1\n"                              \
  "smlal v17.8h, %[v3].8b, v4.8b\n"     /*b+=r2_01234*w[1][0]*/ \
  "smlal v18.8h, %[v1].8b, v8.8b\n"     /*c+=r0_1234*w[0][1]*/  \
  "smlal v19.8h, %[v1].8b, v9.8b\n"     /*d+=r1_1234*w[0][1]*/  \
  "saddw v20.4s, v20.4s, v16.4h\n"                              \
  "saddw v21.4s, v21.4s, v17.4h\n"                              \
  "smull v22.8h, %[v7].8b, v10.8b\n"    /*e=r2_1234*w[2][1]*/   \
  "smull v24.8h, %[v7].8b, v11.8b\n"    /*f=r3_1234*w[2][1]*/   \
  "ext v8.8b, v0.8b, v1.8b, #2\n"                               \
  "ext v11.8b, v6.8b, v7.8b, #2\n"                              \
  "saddw2 v25.4s, v25.4s, v16.8h\n"      /*out0 += a*/          \
  "saddw2 v26.4s, v26.4s, v17.8h\n"      /*out1 += b*/          \
  "smull v16.8h, %[v4].8b, v9.8b\n"      /*a=r1_00123*w[1][1]*/ \
  "smull v17.8h, %[v4].8b, v10.8b\n"     /*b=r2_00123*w[1][1]*/ \
  "ext v9.8b, v2.8b, v3.8b, #2\n"                               \
  "ext v10.8b, v4.8b, v5.8b, #2\n"                              \
  "saddw v20.4s, v20.4s, v18.4h\n"                              \
  "saddw v21.4s, v21.4s, v19.4h\n"                              \
  "smlal v22.8h, %[v2].8b, v8.8b\n"      /*e+=r0_2345*w[0][2]*/\
  "smlal v24.8h, %[v2].8b, v9.8b\n"      /*f+=r1_2345*w[0][2]*/\
  "smlal v16.8h, %[v8].8b, v10.8b\n"     /*a+=r2_2345*w[2][2]*/\
  "smlal v17.8h, %[v8].8b, v11.8b\n"     /*b+=r3_2345*w[2][2]*/\
  "saddw2 v25.4s, v25.4s, v18.8h\n"      /*out0 += c*/          \
  "saddw2 v26.4s, v26.4s, v19.8h\n"      /*out1 += d*/          \
  "smull v18.8h, %[v5].8b, v3.8b\n"      /*c=r1_2345 * w[1][2]*/\
  "smull v19.8h, %[v5].8b, v5.8b\n"      /*d=r2_2345 * w[1][2]*/\
  "saddw v20.4s, v20.4s, v22.4h\n"                              \
  "saddw v21.4s, v21.4s, v24.4h\n"                              \
  "saddw2 v25.4s, v25.4s, v22.8h\n"                             \
  "saddw2 v26.4s, v26.4s, v24.8h\n"                             \
  "saddw v20.4s, v20.4s, v16.4h\n"       /*out0 += e*/          \
  "saddw v21.4s, v21.4s, v17.4h\n"       /*out1 += f*/          \
  "saddw2 v25.4s, v25.4s, v16.8h\n"       /*out1 += a*/         \
  "saddw2 v26.4s, v26.4s, v17.8h\n"       /*out1 += b*/         \
  "saddw v20.4s, v20.4s, v18.4h\n"                              \
  "saddw v21.4s, v21.4s, v19.4h\n"                              \
  "saddw2 v25.4s, v25.4s, v18.8h\n"      /*out1 += c*/          \
  "saddw2 v26.4s, v26.4s, v19.8h\n"      /*out1 += d*/

#define RIGHT_RESULT_INT8_FP32_OUT_S1                           \
  INT8_RIGHT_COMPUTE_S1                                         \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  "ld1 {v0.4s}, [%[ptr_out0]], #16\n"                           \
  "ld1 {v2.4s}, [%[ptr_out1]], #16\n"                           \
  "ld1 {v1.4s}, [%[ptr_out0]]\n"                                \
  "ld1 {v3.4s}, [%[ptr_out1]]\n"                                \
  "sub %[ptr_out0], %[ptr_out0], #16\n"                         \
  "sub %[ptr_out1], %[ptr_out1], #16\n"                         \
  "bif v12.16b, v0.16b, v27.16b\n"                              \
  "bif v13.16b, v2.16b, v27.16b\n"                              \
  "bif v14.16b, v1.16b, v28.16b\n"                              \
  "bif v15.16b, v3.16b, v28.16b\n"                              \
  "stp q12, q14, [%[ptr_out0]], #32\n"                          \
  "stp q13, q15, [%[ptr_out1]], #32\n"

#define RIGHT_RESULT_INT8_INT8_OUT_S1                           \
  INT8_RIGHT_COMPUTE_S1                                         \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  FP32_TO_INT8_S1                                               \
  "ld1 {v0.8b}, [%[rmask1]]\n"                                  \
  "ld1 {v1.8b}, [%[ptr_out0]]\n"                                \
  "ld1 {v2.8b}, [%[ptr_out1]]\n"                                \
  "bif v16.8b, v1.8b, v0.8b\n"                                  \
  "bif v17.8b, v2.9b, v0.8b\n"                                  \
  "st1 {v16.8b}, [%[ptr_out0]], #8\n"                           \
  "st1 {v17.8b}, [%[ptr_out1]], #8\n"

#else
#define INT8_INIT_S1                                            \
  "pld [%[din_ptr0]]                    @ preload data\n"       \
  "pld [%[din_ptr1]]                    @ preload data\n"       \
  "pld [%[din_ptr2]]                    @ preload data\n"       \
  "pld [%[din_ptr3]]                    @ preload data\n"       \
  "vld1.8 {d0-d1}, [%[wei_ptr]]\n"                              \
  "vld1.8 {d12-d13}, [%[din_ptr0]]\n"                           \
  "vdup.s8 d2, d0[0]                    @ w00\n"                \
  "vdup.s8 d5, d0[3]                    @ w10\n"                \
  "vld1.8 {d14-d15}, [%[din_ptr1]]\n"                           \
  "vdup.s8 d8, d0[6]                    @ w20\n"                \
  "vdup.s8 d3, d0[1]                    @ w01\n"                \
  "vdup.s8 d6, d0[4]                    @ w11\n"                \
  "vdup.s8 d9, d0[7]                    @ w21\n"                \
  "vmov.u32 d11, #0                      @ zero\n"

#define INT8_LEFT_COMPUTE_S1                                    \
  "vmull.s8 q10, d12, d3                @ a=r0_0123*w01\n"      \
  "vmull.s8 q11, d14, d3                @ b=r1_0123*w01\n"      \
  "vext.8 d16, d11, d12, #7             @ 0123->00123\n"        \
  "vext.8 d18, d11, d14, #7             @ 0123->00123\n"        \
  "vext.8 d17, d12, d13, #1             @ 0123->1234\n"         \
  "vext.8 d19, d14, d15, #1             @ 0123->1234\n"         \
  "vmlal.s8 q10, d16, d2                @ a+=r0_0012*w00\n"     \
  "vmlal.s8 q11, d18, d2                @ b+=r1_0012*w00\n"     \
  "vdup.s8 d4, d0[2]                    @ w02\n"                \
  "vdup.s8 d7, d0[5]                    @ w12\n"                \
  "vdup.s8 d10, d1[0]                   @ w22\n"                \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d14, d6                @ a=r1_0123*w11\n"      \
  "vmull.s8 q11, d19, d4                @ b=r1_1234*w02\n"      \
  "vld1.8 {d12-d13}, [%[din_ptr2]]\n"                           \
  "vld1.8 {d14-d15}, [%[din_ptr3]]\n"                           \
  "add %[din_ptr0], #7\n"                                       \
  "add %[din_ptr1], #7\n"                                       \
  "vmlal.s8 q10, d17, d4                @ a+=r0_1234*w02\n"     \
  "vmlal.s8 q11, d12, d6                @ b+=r2_0123*w11\n"     \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vext.8 d16, d11, d12, #7             @ r2_0123->00123\n"     \
  "vext.8 d17, d12, d13, #1             @ r2_0123->1234\n"      \
  "vmull.s8 q10, d18, d5                @ a=r1_0012*w10\n"      \
  "vmull.s8 q11, d16, d5                @ b=r2_0012*w10\n"      \
  "vext.8 d18, d11, d14, #7             @ r3_0123->00123\n"     \
  "vmlal.s8 q10, d19, d7                @ a+=r1_1234*w12\n"     \
  "vmlal.s8 q11, d17, d7                @ b+=r2_1234*w12\n"     \
  "vext.8 d19, d14, d15, #1             @ r3_0123->1234\n"      \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d12, d9                @ a=r2_0123*w21\n"      \
  "vmull.s8 q11, d14, d9                @ b=r3_0123*w21\n"      \
  "add %[din_ptr2], #7\n"                                       \
  "add %[din_ptr3], #7\n"                                       \
  "vmlal.s8 q10, d16, d8                @ a+=r2_0012*w20\n"     \
  "vmlal.s8 q11, d18, d8                @ b+=r3_0012*w20\n"     \
  "vld1.8 {d12-d13}, [%[din_ptr0]]\n"                           \
  "vld1.8 {d14-d15}, [%[din_ptr1]]\n"                           \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d17, d10               @ a=r2_1234*w22\n"      \
  "vmull.s8 q11, d19, d10               @ b=r3_1234*w22\n"      \
  "add %[din_ptr0], #8\n"                                       \
  "add %[din_ptr1], #8\n"                                       \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"

#define TRANS_INT32_TO_FP32_S1                                  \
  "vld1.32 {d0-d1}, [%[scale]]           @ load scale\n"        \
  "vcvt.f32.s32 q8, q12                  @ int32-> fp32\n"      \
  "vcvt.f32.s32 q9, q13                  @ int32-> fp32\n"      \
  "vcvt.f32.s32 q10, q14                 @ int32-> fp32\n"      \
  "vcvt.f32.s32 q11, q15                 @ int32-> fp32\n"      \
  "vdup.32 q12, %[bias_val]              @ load bias\n"         \
  "vdup.32 q13, %[bias_val]              @ load bias\n"         \
  "vdup.32 q14, %[bias_val]              @ load bias\n"         \
  "vdup.32 q15, %[bias_val]              @ load bias\n"         \
  "cmp  %[is_relu], #0                   @ skip relu\n"         \
  "vmla.f32  q12, q8, q0                 @ mul scale\n"         \
  "vmla.f32  q13, q9, q0                 @ mul scale\n"         \
  "vmla.f32  q14, q10, q0                @ mul scale\n"         \
  "vmla.f32  q15, q11, q0                @ mul scale\n"

#define S1_RELU                                                 \
  "beq 0f                                @ no act\n"            \
  "cmp  %[is_relu], #1                   @ skip relu\n"         \
  "vmov.f32 q0, #0\n"                                           \
  "bne 1f                                @ no relu\n"           \
  "vmax.f32 q12, q12, q0                 @ do relu\n"           \
  "vmax.f32 q13, q13, q0                 @ do relu\n"           \
  "vmax.f32 q14, q14, q0                 @ do relu\n"           \
  "vmax.f32 q15, q15, q0                 @ do relu\n"           \
  "b   0f\n"
              
#define S1_RELU6                                                \
  "1:  \n"                                                      \
  "cmp  %[is_relu], #2                   @ skip relu\n"         \
  "vld1.32 {d16-d17}, [%[alpha]]         @ load alpha\n"        \
  "bne 2f                                @ no relu6\n"          \
  "vmax.f32 q12, q12, q0                 @ do relu\n"           \
  "vmax.f32 q13, q13, q0                 @ do relu\n"           \
  "vmax.f32 q14, q14, q0                 @ do relu\n"           \
  "vmax.f32 q15, q15, q0                 @ do relu\n"           \
  "vmin.f32 q12, q12, q8                 @ do relu\n"           \
  "vmin.f32 q13, q13, q8                 @ do relu\n"           \
  "vmin.f32 q14, q14, q8                 @ do relu\n"           \
  "vmin.f32 q15, q15, q8                 @ do relu\n"           \
  "b   0f\n"

#define S1_LEAKY_RELU                                           \
  "2:  \n"                                                      \
  "vcge.f32 q9, q12, q0                  @ vcgeq_u32\n"         \
  "vmul.f32 q10, q12, q8                 @ vmulq_f32\n"         \
  "vcge.f32 q11, q13, q0                 @ vcgeq_u32\n"         \
  "vbif q12, q10, q9                     @ choose\n"            \
  "vmul.f32 q9, q13, q8                  @ vmulq_f32\n"         \
  "vcge.f32 q10, q14, q0                 @ vcgeq_u32\n"         \
  "vbif q13, q9, q11                     @ choose\n"            \
  "vmul.f32 q11, q14, q8                 @ vmulq_f32\n"         \
  "vcge.f32 q9, q15, q0                  @ vcgeq_u32\n"         \
  "vbif q14, q11, q10                    @ choose\n"            \
  "vmul.f32 q10, q15, q8                 @ vmulq_f32\n"         \
  "vbif q15, q10, q9                     @ choose\n"            \
  "0:   \n"

#define FP32_TO_INT8_S1                                         \
  "vmov.f32 q0, #0\n"                                           \
  "vmov.f32 q8, #-0.5\n"                                        \
  "vmov.f32 q9, #0.5\n"                                         \
  "vcgt.f32 q10, q12, q0\n"                                     \
  "vcgt.f32 q11, q13, q0\n"                                     \
  "vbif.f32 q9, q8, q10\n"                                      \
  "vadd.f32 q12, q12, q9\n"                                     \
  "vmov.f32 q9, #0.5\n"                                         \
  "vcgt.f32 q10, q14, q0\n"                                     \
  "vbif.f32 q9, q8, q11\n"                                      \
  "vadd.f32 q13, q13, q9\n"                                     \
  "vmov.f32 q9, #0.5\n"                                         \
  "vcgt.f32 q11, q15, q0\n"                                     \
  "vbif.f32 q9, q8, q10\n"                                      \
  /* data >= -127 */                                            \
  "vld1.32 {d0-d1}, [%[vmax]]\n"                                \
  "vadd.f32 q14, q14, q9\n"                                     \
  "vmov.f32 q9, #0.5\n"                                         \
  "vbif.f32 q9, q8, q11\n"                                      \
  "vadd.f32 q15, q15, q9\n"                                     \
  "vcge.f32 q8, q12, q0\n"                                      \
  "vcge.f32 q9, q13, q0\n"                                      \
  "vcge.f32 q10, q14, q0\n"                                     \
  "vcge.f32 q11, q15, q0\n"                                     \
  "vbif.f32 q12, q0, q8\n"                                      \
  "vbif.f32 q13, q0, q9\n"                                      \
  "vbif.f32 q14, q0, q10\n"                                     \
  "vbif.f32 q15, q0, q11\n"                                     \
  "vcvt.s32.f32 q8, q12                     @ fp32 to int32\n"  \
  "vcvt.s32.f32 q9, q13                     @ fp32 to int32\n"  \
  "vcvt.s32.f32 q10, q14                    @ fp32 to int32\n"  \
  "vcvt.s32.f32 q11, q15                    @ fp32 to int32\n"  \
  "vqmovn.s32 d24, q8                       @ int32 to int16\n" \
  "vqmovn.s32 d25, q9                       @ int32 to int16\n" \
  "vqmovn.s32 d26, q10                      @ int32 to int16\n" \
  "vqmovn.s32 d27, q11                      @ int32 to int16\n" \
  "vqmovn.s16 d16, q12                      @ int16 to int8\n"  \
  "vqmovn.s16 d17, q13                      @ int16 to int8\n"  \

#define LEFT_RESULT_INT8_FP32_OUT_S1                            \
  INT8_LEFT_COMPUTE_S1                                          \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  "cmp  %w[cnt], #1\n"                                          \
  "stp q12, q13, [%[ptr_out0]!\n"                               \
  "stp q14, q15, [%[ptr_out1]!\n"                               \
  "blt 4f\n"

#define LEFT_RESULT_INT8_INT8_OUT_S1                            \
  INT8_LEFT_COMPUTE_S1                                          \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  FP32_TO_INT8_S1                                               \
  "cmp  %w[cnt], #1\n"                                          \
  "vst1.32 d16, [%[ptr_out0]!\n"                                \
  "vst1.32 d17, [%[ptr_out1]!\n"                                \
  "blt 4f\n"

#define INT8_MID_COMPUTE_S1                                     \
  "3:\n"                                                        \
  "vmull.s8 q10, d12, d2                @ a=r0_0123*w00\n"      \
  "vmull.s8 q11, d14, d2                @ b=r1_0123*w00\n"      \
  "vext.8 d16, d12, d13, #1             @ 0123->1234\n"        \
  "vext.8 d18, d14, d15, #1             @ 0123->1234\n"        \
  "vext.8 d17, d12, d13, #2             @ 0123->2345\n"         \
  "vext.8 d19, d14, d15, #2             @ 0123->2345\n"         \
  "vmlal.s8 q10, d16, d3                @ a+=r0_1234*w01\n"     \
  "vmlal.s8 q11, d18, d3                @ b+=r1_1234*w01\n"     \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d14, d5                @ a=r1_0123*w10\n"      \
  "vmull.s8 q11, d19, d4                @ b=r1_2345*w02\n"      \
  "vld1.8 {d12-d13}, [%[din_ptr2]]\n"                           \
  "vld1.8 {d14-d15}, [%[din_ptr3]]\n"                           \
  "vmlal.s8 q10, d17, d4                @ a+=r0_2345*w02\n"     \
  "vmlal.s8 q11, d12, d5                @ b+=r2_0123*w10\n"     \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vext.8 d16, d12, d14, #1             @ r2_0123->12345\n"     \
  "vext.8 d17, d12, d13, #2             @ r2_0123->2345\n"      \
  "vmull.s8 q10, d18, d6                @ a=r1_1234*w11\n"      \
  "vmull.s8 q11, d16, d6                @ b=r2_1234*w11\n"      \
  "vext.8 d18, d14, d15, #1             @ r3_0123->12345\n"     \
  "vmlal.s8 q10, d19, d7                @ a+=r1_2345*w12\n"     \
  "vmlal.s8 q11, d17, d7                @ b+=r2_2345*w12\n"     \
  "vext.8 d19, d14, d15, #2             @ r3_0123->2345\n"      \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d12, d8                @ a=r2_0123*w20\n"      \
  "vmull.s8 q11, d14, d8                @ b=r3_0123*w20\n"      \
  "add %[din_ptr2], #8\n"                                       \
  "add %[din_ptr3], #8\n"                                       \
  "vmlal.s8 q10, d16, d9                @ a+=r2_1234*w21\n"     \
  "vmlal.s8 q11, d18, d9                @ b+=r3_1234*w21\n"     \
  "vld1.8 {d12-d13}, [%[din_ptr0]]\n"                           \
  "vld1.8 {d14-d15}, [%[din_ptr1]]\n"                           \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d17, d10               @ a=r2_1234*w22\n"      \
  "vmull.s8 q11, d19, d10               @ b=r3_1234*w22\n"      \
  "add %[din_ptr0], #8\n"                                       \
  "add %[din_ptr1], #8\n"                                       \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"
#define MID_RESULT_INT8_FP32_OUT_S1                             \
  INT8_MID_COMPUTE_S1                                           \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  "subs %[cnt], #1\n"                                           \
  "stp q12, q13, [%[ptr_out0]!\n"                               \
  "stp q14, q15, [%[ptr_out1]!\n"                               \
  "bne 3b\n"

#define MID_RESULT_INT8_INT8_OUT_S1                             \
  INT8_MID_COMPUTE_S1                                           \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  FP32_TO_INT8_S1                                               \
  "subs %[cnt], #1\n"                                           \
  "vst1.32 d16, [%[ptr_out0]!\n"                                \
  "vst1.32 d17, [%[ptr_out1]!\n"                                \
  "bne 3b\n"

#define INT8_RIGHT_COMPUTE_S1                                   \
  "4:\n"                                                        \
  "ld1 {d16-d17}, [%[vmask]]\n"                                 \
  "vbif.8 d12, d11, d16\n"                                      \
  "vbif.8 d14, d11, d16\n"                                      \
  "vbif.8 d13, d11, d17\n"                                      \
  "vbif.8 d15, d11, d17\n"                                      \
  "vmull.s8 q10, d12, d2                @ a=r0_0123*w00\n"      \
  "vmull.s8 q11, d14, d2                @ b=r1_0123*w00\n"      \
  "vext.8 d16, d12, d13, #1             @ 0123->1234\n"        \
  "vext.8 d18, d14, d15, #1             @ 0123->1234\n"        \
  "vext.8 d17, d12, d13, #2             @ 0123->2345\n"         \
  "vext.8 d19, d14, d15, #2             @ 0123->2345\n"         \
  "vmlal.s8 q10, d16, d3                @ a+=r0_1234*w01\n"     \
  "vmlal.s8 q11, d18, d3                @ b+=r1_1234*w01\n"     \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d14, d5                @ a=r1_0123*w10\n"      \
  "vmull.s8 q11, d19, d4                @ b=r1_2345*w02\n"      \
  "vld1.8 {d12-d13}, [%[din_ptr2]]\n"                           \
  "vld1.8 {d14-d15}, [%[din_ptr3]]\n"                           \
  "vmlal.s8 q10, d17, d4                @ a+=r0_2345*w02\n"     \
  "ld1 {d16-d17}, [%[vmask]]\n"                                 \
  "vbif.8 d12, d11, d16\n"                                      \
  "vbif.8 d14, d11, d16\n"                                      \
  "vbif.8 d13, d11, d17\n"                                      \
  "vbif.8 d15, d11, d17\n"                                      \
  "vmlal.s8 q11, d12, d5                @ b+=r2_0123*w10\n"     \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vext.8 d16, d12, d14, #1             @ r2_0123->12345\n"     \
  "vext.8 d17, d12, d13, #2             @ r2_0123->2345\n"      \
  "vmull.s8 q10, d18, d6                @ a=r1_1234*w11\n"      \
  "vmull.s8 q11, d16, d6                @ b=r2_1234*w11\n"      \
  "vext.8 d18, d14, d15, #1             @ r3_0123->12345\n"     \
  "vmlal.s8 q10, d19, d7                @ a+=r1_2345*w12\n"     \
  "vmlal.s8 q11, d17, d7                @ b+=r2_2345*w12\n"     \
  "vext.8 d19, d14, d15, #2             @ r3_0123->2345\n"      \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d12, d8                @ a=r2_0123*w20\n"      \
  "vmull.s8 q11, d14, d8                @ b=r3_0123*w20\n"      \
  "add %[din_ptr2], #8\n"                                       \
  "add %[din_ptr3], #8\n"                                       \
  "vmlal.s8 q10, d16, d9                @ a+=r2_1234*w21\n"     \
  "vmlal.s8 q11, d18, d9                @ b+=r3_1234*w21\n"     \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"                \
  "vmull.s8 q10, d17, d10               @ a=r2_1234*w22\n"      \
  "vmull.s8 q11, d19, d10               @ b=r3_1234*w22\n"      \
  "vaddw.s16 q12, q12, d20              @ out0+=a\n"            \
  "vaddw.s16 q13, q13, d21              @ add\n"                \
  "vaddw.s16 q14, q14, d22              @ out0+=b\n"            \
  "vaddw.s16 q15, q15, d23              @ add\n"

#define RIGHT_RESULT_INT8_FP32_OUT_S1                           \
  INT8_RIGHT_COMPUTE_S1                                         \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  "ld1 {d16-d19}, [%[rmask]]\n"                                 \
  "ld1 {d0-d1}, [%[ptr_out0]], #16\n"                           \
  "ld1 {d4-d5}, [%[ptr_out1]], #16\n"                           \
  "ld1 {d2-d3}, [%[ptr_out0]]\n"                                \
  "ld1 {d6-d7}, [%[ptr_out1]]\n"                                \
  "sub %[ptr_out0], #16\n"                                      \
  "sub %[ptr_out1], #16\n"                                      \
  "vbif q12, q0, q8\n"                                          \
  "vbif q14, q2, q8\n"                                          \
  "vbif q13, q1, q9\n"                                          \
  "vbif q15, q3, q9\n"                                          \
  "stp q12, q13, [%[ptr_out0]!\n"                               \
  "stp q14, q15, [%[ptr_out1]!\n"

#define RIGHT_RESULT_INT8_INT8_OUT_S1                           \
  INT8_RIGHT_COMPUTE_S1                                         \
  TRANS_INT32_TO_FP32_S1                                        \
  S1_RELU                                                       \
  S1_RELU6                                                      \
  S1_LEAKY_RELU                                                 \
  FP32_TO_INT8_S1                                               \
  "ld1 {d0}, [%[rmask1]]\n"                                     \
  "ld1 {d1}, [%[ptr_out0]]\n"                                   \
  "ld1 {d2}, [%[ptr_out1]]\n"                                   \
  "vbif d16, d1, d0\n"                                          \
  "vbif d17, d2, d0\n"                                          \
  "vst1.32 d16, [%[ptr_out0]!\n"                                \
  "vst1.32 d17, [%[ptr_out1]!\n"

#endif
// clang-format on

template <typename Dtype>
void conv3x3s1p1_kernel(const int8_t* din_ptr0,
                        const int8_t* din_ptr1,
                        const int8_t* din_ptr2,
                        const int8_t* din_ptr3,
                        Dtype* doutr0,
                        Dtype* doutr1,
                        const int8_t* weights,
                        const float* scale,
                        const float* bias,
                        unsigned int *vmask_ptr,
                        unsigned int *rmask_ptr,
                        unsigned char *rmask_ptr1,
                        int flag_act,
                        const float* alpha);

template <>
void conv3x3s1p1_kernel(const int8_t* din_ptr0,
                        const int8_t* din_ptr1,
                        const int8_t* din_ptr2,
                        const int8_t* din_ptr3,
                        float* doutr0,
                        float* doutr1,
                        const int8_t* weights,
                        const float* scale,
                        const float* bias,
                        unsigned int *vmask_ptr,
                        unsigned int *rmask_ptr,
                        unsigned char *rmask_ptr1,
                        int flag_act,
                        const float* alpha){

}

template <>
void conv3x3s1p1_kernel(const int8_t* din_ptr0,
                        const int8_t* din_ptr1,
                        const int8_t* din_ptr2,
                        const int8_t* din_ptr3,
                        int8_t* doutr0,
                        int8_t* doutr1,
                        const int8_t* weights,
                        const float* scale,
                        const float* bias,
                        unsigned int *rmask_ptr,
                        unsigned int *vmask_ptr,
                        int flag_act,
                        const float* alpha){

}
template <typename Dtype>
void conv_depthwise_3x3s1p1_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx) {
  //! pad is done implicit
  const int8_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  //! for 4x10 convolution window
  const unsigned char right_pad_idx[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0};
  const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t *zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  Dtype *write_ptr = reinterpret_cast<Dtype*>(zero_ptr + win);

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 9;

  int tile_w = wout >> 3;
  int remain = wout % 7;
  int cnt_col = tile_w - 1;

  unsigned int size_pad_right = (unsigned int)(9 + (tile_w << 3) - win);

  if (remain == 0 && size_pad_right == 9) {
    size_pad_right = 7;
    cnt_col -= 1;
    remain = 8;
  } else if (remain == 0 && size_pad_right == 10) {
    size_pad_right = 6;
    cnt_col -= 1;
    remain = 8;
  }

  uint8x16_t vmask_rp = vcgeq_u8(vld1q_u8(right_pad_idx), vdupq_n_u8(size_pad_right));

  uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(remain), vld1q_u32(right_pad_rst));
  uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(remain), vld1q_u32(right_pad_rst + 4));
  uint8x8_t vmask_result3 = vcgt_u8(vdup_n_u8(remain), vld1_u8(right_pad_rst));

  unsigned int vmask[16];
  vst1q_u8(vmask, vmask_rp);

  unsigned int rmask[8];
  vst1q_u32(rmask, vmask_result1);
  vst1q_u32(rmask + 4, vmask_result1);

  uint8_t rmask_1[8];
  vst1_u8(rmask_1, vmask_result3);

  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }

  for (int n = 0; n < num; ++n) {
      const int8_t *din_batch = din + n * chin * size_in_channel;
      Dtype *dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chin; c++) {
          const int8_t *din_ptr = din_batch + c * size_in_channel;
          Dtype *dout_ptr = dout_batch + c * size_out_channel;
          float bias_val = flag_bias ? bias[c] : 0.0;
          const int8_t* wei_ptr = weights + c * w_stride;
          const float* scale_ptr = scale + c;
          float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
          const int8_t *dr0 = din_ptr;
          const int8_t *dr1 = dr0 + win;
          const int8_t *dr2 = dr1 + win;
          const int8_t *dr3 = dr2 + win;
          Dtype *doutr0 = dout_ptr;
          Dtype *doutr1 = dout_ptr + wout;
          for (int h = 0; h < hout; h += 2){
              const int8_t *din_ptr0 = dr0;
              const int8_t *din_ptr1 = dr1;
              const int8_t *din_ptr2 = dr2;
              const int8_t *din_ptr3 = dr3;
              if (h == 0){
                  din_ptr0 = zero_ptr;
                  din_ptr1 = dr0;
                  din_ptr2 = dr1;
                  din_ptr3 = dr2;
                  dr0 = dr1;
                  dr1 = dr2;
                  dr2 = dr3;
                  dr3 = dr2 + win;
              } else {
                  dr0 = dr2;
                  dr1 = dr3;
                  dr2 = dr1 + win;
                  dr3 = dr2 + win;
              }
              //! process bottom pad
              if (h + 2 > hout) {
                  doutr1 = write_ptr;
              }
              if (h + 3 > hin) {
                  switch (h + 3 - hin) {
                      case 3:
                          din_ptr1 = zero_ptr;
                      case 2:
                          din_ptr2 = zero_ptr;
                      case 1:
                          din_ptr3 = zero_ptr;
                      default:
                          break;
                  }
              }
              unsigned int *vmask_ptr = vmask;
              unsigned int *rmask_ptr = rmask;
              // asm
              conv3x3s1p1_kernel<Dtype>(din_ptr0,
                                        din_ptr1,
                                        din_ptr2,
                                        din_ptr3,
                                        doutr0,
                                        doutr1,
                                        wei_ptr,
                                        scale_ptr,
                                        vbias,
                                        vmask_ptr,
                                        rmask_ptr,
                                        rmask_1,
                                        flag_act,
                                        alpha);

              din_ptr += 2 * win;
              dout_ptr += 2 * wout;
          }
      }
  }
}

template <typename Dtype>
void conv3x3s1p0_kernel(const int8_t* din_ptr0,
                        const int8_t* din_ptr1,
                        const int8_t* din_ptr2,
                        const int8_t* din_ptr3,
                        Dtype* doutr0,
                        Dtype* doutr1,
                        const int8_t* weights,
                        const float* scale,
                        const float* bias,
                        unsigned int *vmask_ptr,
                        unsigned int *rmask_ptr,
                        unsigned char *rmask_ptr1,
                        int flag_act,
                        const float* alpha);
// clang-format off
#ifdef __aarch64__
#else
#endif
// clang-format on

template <>
void conv3x3s1p0_kernel(const int8_t* din_ptr0,
                        const int8_t* din_ptr1,
                        const int8_t* din_ptr2,
                        const int8_t* din_ptr3,
                        float* doutr0,
                        float* doutr1,
                        const int8_t* weights,
                        const float* scale,
                        const float* bias,
                        unsigned int *vmask_ptr,
                        unsigned int *rmask_ptr,
                        unsigned char *rmask_ptr1,
                        int flag_act,
                        const float* alpha){

}

template <>
void conv3x3s1p0_kernel(const int8_t* din_ptr0,
                        const int8_t* din_ptr1,
                        const int8_t* din_ptr2,
                        const int8_t* din_ptr3,
                        int8_t* doutr0,
                        int8_t* doutr1,
                        const int8_t* weights,
                        const float* scale,
                        const float* bias,
                        unsigned int *vmask_ptr,
                        unsigned int *rmask_ptr,
                        unsigned char *rmask_ptr1,
                        int flag_act,
                        const float* alpha){

}
template <typename Dtype>
void conv_depthwise_3x3s1p0_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx) {
  //! pad is done implicit
  const int8_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  //! for 4x10 convolution window
  const unsigned char right_pad_idx[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0};
  const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t *zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  Dtype *write_ptr = reinterpret_cast<Dtype*>(zero_ptr + win);

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 9;

  int tile_w = wout >> 3;
  int remain = wout % 7;
  int cnt_col = tile_w - 1;

  unsigned int size_pad_right = (unsigned int)(10 + (tile_w << 3) - win);

  if (remain == 0 && size_pad_right == 10) {
    size_pad_right = 6;
    cnt_col -= 1;
    remain = 8;
  }

  uint8x16_t vmask_rp = vcgeq_u8(vld1q_u8(right_pad_idx), vdupq_n_u8(size_pad_right));

  uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(remain), vld1q_u32(right_pad_rst));
  uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(remain), vld1q_u32(right_pad_rst + 4));
  uint8x8_t vmask_result3 = vcgt_u8(vdup_n_u8(remain), vld1_u8(right_pad_rst));

  unsigned int vmask[16];
  vst1q_u8(vmask, vmask_rp);

  unsigned int rmask[8];
  vst1q_u32(rmask, vmask_result1);
  vst1q_u32(rmask + 4, vmask_result1);

  uint8_t rmask_1[8];
  vst1_u8(rmask_1, vmask_result3);

  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    }
  }

  for (int n = 0; n < num; ++n) {
      const int8_t *din_batch = din + n * chin * size_in_channel;
      Dtype *dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chin; c++) {
          const int8_t *din_ptr = din_batch + c * size_in_channel;
          Dtype *dout_ptr = dout_batch + c * size_out_channel;
          float bias_val = flag_bias ? bias[c] : 0.0;
          const int8_t* wei_ptr = weights + c * w_stride;
          const float* scale_ptr = scale + c;
          float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
          const int8_t *dr0 = din_ptr;
          const int8_t *dr1 = dr0 + win;
          const int8_t *dr2 = dr1 + win;
          const int8_t *dr3 = dr2 + win;
          Dtype *doutr0 = dout_ptr;
          Dtype *doutr1 = dout_ptr + wout;
          for (int h = 0; h < hout; h += 2){
              const int8_t *din_ptr0 = dr0;
              const int8_t *din_ptr1 = dr1;
              const int8_t *din_ptr2 = dr2;
              const int8_t *din_ptr3 = dr3;
              //! process bottom pad
              if (h + 2 > hout) {
                  doutr1 = write_ptr;
              }
              if (h + 3 > hin) {
                  switch (h + 3 - hin) {
                      case 3:
                          din_ptr1 = zero_ptr;
                      case 2:
                          din_ptr2 = zero_ptr;
                      case 1:
                          din_ptr3 = zero_ptr;
                      default:
                          break;
                  }
              }
              unsigned int *vmask_ptr = vmask;
              unsigned int *rmask_ptr = rmask;
              // asm
              conv3x3s1p0_kernel<Dtype>(din_ptr0,
                                        din_ptr1,
                                        din_ptr2,
                                        din_ptr3,
                                        doutr0,
                                        doutr1,
                                        wei_ptr,
                                        scale_ptr,
                                        vbias,
                                        vmask_ptr,
                                        rmask_ptr,
                                        rmask_1,
                                        flag_act,
                                        alpha);

              dr0 = dr2;
              dr1 = dr3;
              dr2 = dr1 + win;
              dr3 = dr2 + win;
              din_ptr += 2 * win;
              dout_ptr += 2 * wout;
          }
      }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle