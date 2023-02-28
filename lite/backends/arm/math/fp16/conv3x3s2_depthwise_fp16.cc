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

#include <arm_neon.h>
#include <utility>
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/fp16/conv3x3_depthwise_fp16.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#ifdef __aarch64__
#define INIT_FP16_S2                                    \
  "PRFM PLDL1KEEP, [%[din_ptr0]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr1]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr2]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr3]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr4]]                    \n" \
  "ld2    {v0.8h, v1.8h}, [%[din_ptr0]], #32        \n" \
  "ld2    {v2.8h, v3.8h}, [%[din_ptr1]], #32        \n" \
  "ld2    {v4.8h, v5.8h}, [%[din_ptr2]], #32        \n" \
  "ld2    {v6.8h, v7.8h}, [%[din_ptr3]], #32        \n" \
  "ld2    {v8.8h, v9.8h}, [%[din_ptr4]], #32        \n"

#define LEFT_COMPUTE_FP16_S2                            \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ext    v10.16b, %[vzero].16b, v1.16b, #14        \n" \
  "fmul   v11.8h, v0.8h, %[wr01].8h                 \n" \
  "fmul   v12.8h, v1.8h, %[wr02].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr00].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v3.16b, #14        \n" \
  "sub    %[din_ptr0], %[din_ptr0], #2              \n" \
  "sub    %[din_ptr1], %[din_ptr1], #2              \n" \
  "fmla   v11.8h, v2.8h, %[wr11].8h                 \n" \
  "fmla   v12.8h, v3.8h, %[wr12].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr10].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v5.16b, #14        \n" \
  "sub    %[din_ptr2], %[din_ptr2], #2              \n" \
  "sub    %[din_ptr3], %[din_ptr3], #2              \n" \
  "fmul   v13.8h, v4.8h, %[wr01].8h                 \n" \
  "fmla   v11.8h, v4.8h, %[wr21].8h                 \n" \
  "fmul   v14.8h, v5.8h, %[wr02].8h                 \n" \
  "fmla   v12.8h, v5.8h, %[wr22].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr00].8h                \n" \
  "fmla   v16.8h, v10.8h, %[wr20].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v7.16b, #14        \n" \
  "sub    %[din_ptr4], %[din_ptr4], #2              \n" \
  "fmla   v13.8h, v6.8h, %[wr11].8h                 \n" \
  "fmla   v14.8h, v7.8h, %[wr12].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr10].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v9.16b, #14        \n" \
  "fadd   v16.8h, v16.8h, v11.8h                    \n" \
  "fadd   v16.8h, v16.8h, v12.8h                    \n" \
  "fmla   v13.8h, v8.8h, %[wr21].8h                 \n" \
  "fmla   v14.8h, v9.8h, %[wr22].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr20].8h                \n" \
  "fadd   v17.8h, v17.8h, v13.8h                    \n" \
  "fadd   v17.8h, v17.8h, v14.8h                    \n" \
  "ld2    {v0.8h, v1.8h}, [%[din_ptr0]], #32        \n" \
  "ld2    {v2.8h, v3.8h}, [%[din_ptr1]], #32        \n" \
  "ld2    {v4.8h, v5.8h}, [%[din_ptr2]], #32        \n" \
  "ld2    {v6.8h, v7.8h}, [%[din_ptr3]], #32        \n" \
  "ld2    {v8.8h, v9.8h}, [%[din_ptr4]], #32        \n"

#define LEFT_RESULT_FP16_S2                             \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    1f                                        \n"

#define LEFT_RESULT_FP16_S2_RELU                        \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    1f                                        \n"

#define LEFT_RESULT_FP16_S2_RELU6                       \
  "ldr    q21, [%[bias_val], #0x10]                 \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    1f                                        \n"

#define LEFT_RESULT_FP16_S2_LEAKY_RELU                  \
  "ldr    q21, [%[bias_val], #0x10]                 \n" \
  "fcmge  v12.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v12.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    1f                                        \n"

#define MID_COMPUTE_FP16_S2                             \
  "2:                                               \n" \
  "ld1    {v15.8h}, [%[din_ptr0]]                   \n" \
  "ld1    {v18.8h}, [%[din_ptr1]]                   \n" \
  "ld1    {v19.8h}, [%[din_ptr2]]                   \n" \
  "ld1    {v20.8h}, [%[din_ptr3]]                   \n" \
  "ld1    {v21.8h}, [%[din_ptr4]]                   \n" \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ext    v10.16b, v0.16b, v15.16b, #2              \n" \
  "fmul   v11.8h, v0.8h, %[wr00].8h                 \n" \
  "fmul   v12.8h, v1.8h, %[wr01].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr02].8h                \n" \
  "ext    v10.16b, v2.16b, v18.16b, #2              \n" \
  "ld2    {v0.8h, v1.8h}, [%[din_ptr0]], #32        \n" \
  "fmla   v11.8h, v2.8h, %[wr10].8h                 \n" \
  "fmla   v12.8h, v3.8h, %[wr11].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr12].8h                \n" \
  "ext    v10.16b, v4.16b, v19.16b, #2              \n" \
  "ld2    {v2.8h, v3.8h}, [%[din_ptr1]], #32        \n" \
  "fmul   v13.8h, v4.8h, %[wr00].8h                 \n" \
  "fmla   v11.8h, v4.8h, %[wr20].8h                 \n" \
  "fmul   v14.8h, v5.8h, %[wr01].8h                 \n" \
  "fmla   v12.8h, v5.8h, %[wr21].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr02].8h                \n" \
  "fmla   v16.8h, v10.8h, %[wr22].8h                \n" \
  "ext    v10.16b, v6.16b, v20.16b, #2              \n" \
  "ld2    {v4.8h, v5.8h}, [%[din_ptr2]], #32        \n" \
  "fmla   v13.8h, v6.8h, %[wr10].8h                 \n" \
  "fmla   v14.8h, v7.8h, %[wr11].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr12].8h                \n" \
  "ext    v10.16b, v8.16b, v21.16b, #2              \n" \
  "ld2    {v6.8h, v7.8h}, [%[din_ptr3]], #32        \n" \
  "fadd   v16.8h, v16.8h, v11.8h                    \n" \
  "fadd   v16.8h, v16.8h, v12.8h                    \n" \
  "fmla   v13.8h, v8.8h, %[wr20].8h                 \n" \
  "fmla   v14.8h, v9.8h, %[wr21].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr22].8h                \n" \
  "ld2    {v8.8h, v9.8h}, [%[din_ptr4]], #32        \n" \
  "fadd   v17.8h, v17.8h, v13.8h                    \n" \
  "fadd   v17.8h, v17.8h, v14.8h                    \n" \
  "subs   %w[cnt], %w[cnt], #1                      \n"

#define MID_RESULT_FP16_S2                              \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "bge    2b                                        \n"

#define MID_RESULT_FP16_S2_RELU                         \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "bge    2b                                        \n"

#define MID_RESULT_FP16_S2_RELU6                        \
  "ldr    q21, [%[bias_val], #0x10]                 \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "bge    2b                                        \n"

#define MID_RESULT_FP16_S2_LEAKY_RELU                   \
  "ldr    q21, [%[bias_val], #0x10]                 \n" \
  "fcmge  v12.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v12.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "bge    2b                                        \n"

#define RIGHT_COMPUTE_FP16_S2                                        \
  "1:                                                     \n"        \
  "cmp    %w[right_st_num], #16                                  \n" \
  "beq    4f                                              \n"        \
  "3:                                                     \n"        \
  "ld1    {v16.8h}, [%[bias_val]]                         \n"        \
  "ld1    {v17.8h}, [%[bias_val]]                         \n"        \
  "ldr    q18, [%[vmask]]                                 \n"        \
  "ldr    q19, [%[vmask], #0x10]                          \n"        \
  "ldr    q20, [%[vmask], #0x20]                          \n"        \
  "sub    %[din_ptr0], %[din_ptr0], %[right_pad_num]      \n"        \
  "sub    %[din_ptr1], %[din_ptr1], %[right_pad_num]      \n"        \
  "sub    %[din_ptr2], %[din_ptr2], %[right_pad_num]      \n"        \
  "sub    %[din_ptr3], %[din_ptr3], %[right_pad_num]      \n"        \
  "sub    %[din_ptr4], %[din_ptr4], %[right_pad_num]      \n"        \
  "sub    %[ptr_out0], %[ptr_out0], %[right_st_num]       \n"        \
  "sub    %[ptr_out1], %[ptr_out1], %[right_st_num]       \n"        \
  "ld2    {v0.8h, v1.8h}, [%[din_ptr0]]                   \n"        \
  "ld2    {v2.8h, v3.8h}, [%[din_ptr1]]                   \n"        \
  "ld2    {v4.8h, v5.8h}, [%[din_ptr2]]                   \n"        \
  "ld2    {v6.8h, v7.8h}, [%[din_ptr3]]                   \n"        \
  "ld2    {v8.8h, v9.8h}, [%[din_ptr4]]                   \n"        \
  "bif    v0.16b, %[vzero].16b, v18.16b                   \n"        \
  "bif    v1.16b, %[vzero].16b, v19.16b                   \n"        \
  "bif    v2.16b, %[vzero].16b, v18.16b                   \n"        \
  "bif    v3.16b, %[vzero].16b, v19.16b                   \n"        \
  "bif    v4.16b, %[vzero].16b, v18.16b                   \n"        \
  "bif    v5.16b, %[vzero].16b, v19.16b                   \n"        \
  "add    %[din_ptr0], %[din_ptr0], #4                    \n"        \
  "add    %[din_ptr1], %[din_ptr1], #4                    \n"        \
  "add    %[din_ptr2], %[din_ptr2], #4                    \n"        \
  "add    %[din_ptr3], %[din_ptr3], #4                    \n"        \
  "add    %[din_ptr4], %[din_ptr4], #4                    \n"        \
  "ld2    {v10.8h, v11.8h}, [%[din_ptr0]]                 \n"        \
  "bif    v10.16b, %[vzero].16b, v20.16b                  \n"        \
  "bif    v6.16b, %[vzero].16b, v18.16b                   \n"        \
  "bif    v7.16b, %[vzero].16b, v19.16b                   \n"        \
  "fmul   v21.8h, v0.8h, %[wr00].8h                       \n"        \
  "fmul   v12.8h, v1.8h, %[wr01].8h                       \n"        \
  "fmla   v16.8h, v10.8h, %[wr02].8h                      \n"        \
  "ld2    {v10.8h, v11.8h}, [%[din_ptr1]]                 \n"        \
  "bif    v10.16b, %[vzero].16b, v20.16b                  \n"        \
  "bif    v8.16b, %[vzero].16b, v18.16b                   \n"        \
  "bif    v9.16b, %[vzero].16b, v19.16b                   \n"        \
  "fmla   v21.8h, v2.8h, %[wr10].8h                       \n"        \
  "fmla   v12.8h, v3.8h, %[wr11].8h                       \n"        \
  "fmla   v16.8h, v10.8h, %[wr12].8h                      \n"        \
  "ld2    {v10.8h, v11.8h}, [%[din_ptr2]]                 \n"        \
  "bif    v10.16b, %[vzero].16b, v20.16b                  \n"        \
  "fmul   v13.8h, v4.8h, %[wr00].8h                       \n"        \
  "fmla   v21.8h, v4.8h, %[wr20].8h                       \n"        \
  "fmul   v14.8h, v5.8h, %[wr01].8h                       \n"        \
  "fmla   v12.8h, v5.8h, %[wr21].8h                       \n"        \
  "fmla   v17.8h, v10.8h, %[wr02].8h                      \n"        \
  "fmla   v16.8h, v10.8h, %[wr22].8h                      \n"        \
  "ld2    {v10.8h, v11.8h}, [%[din_ptr3]]                 \n"        \
  "bif    v10.16b, %[vzero].16b, v20.16b                  \n"        \
  "fmla   v13.8h, v6.8h, %[wr10].8h                       \n"        \
  "fmla   v14.8h, v7.8h, %[wr11].8h                       \n"        \
  "fmla   v17.8h, v10.8h, %[wr12].8h                      \n"        \
  "ld2    {v10.8h, v11.8h}, [%[din_ptr4]]                 \n"        \
  "bif    v10.16b, %[vzero].16b, v20.16b                  \n"        \
  "fadd   v16.8h, v16.8h, v21.8h                          \n"        \
  "fadd   v16.8h, v16.8h, v12.8h                          \n"        \
  "fmla   v13.8h, v8.8h, %[wr20].8h                       \n"        \
  "fmla   v14.8h, v9.8h, %[wr21].8h                       \n"        \
  "fmla   v17.8h, v10.8h, %[wr22].8h                      \n"        \
  "fadd   v17.8h, v17.8h, v13.8h                          \n"        \
  "fadd   v17.8h, v17.8h, v14.8h                          \n"

#define RIGHT_RESULT_FP16_S2_RELU                       \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "4:                                               \n"

#define RIGHT_RESULT_FP16_S2                            \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "4:                                               \n"

#define RIGHT_RESULT_FP16_S2_RELU6                      \
  "ldr    q21, [%[bias_val], #0x10]                 \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "4:                                               \n"

#define RIGHT_RESULT_FP16_S2_LEAKY_RELU                 \
  "ldr    q21, [%[bias_val], #0x10]                 \n" \
  "fcmge  v12.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v12.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "4:                                               \n"

#define RIGHT_COMPUTE_FP16_S2P1_SMALL                   \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ldr    q18, [%[vmask]]                           \n" \
  "ldr    q19, [%[vmask], #0x10]                    \n" \
  "ldr    q20, [%[rmask]]                           \n" \
  "bif    v0.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v1.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v2.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v3.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v4.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v5.16b, %[vzero].16b, v19.16b             \n" \
  "ext    v10.16b, %[vzero].16b, v1.16b, #14        \n" \
  "bif    v6.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v7.16b, %[vzero].16b, v19.16b             \n" \
  "fmul   v11.8h, v0.8h, %[wr01].8h                 \n" \
  "fmul   v12.8h, v1.8h, %[wr02].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr00].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v3.16b, #14        \n" \
  "bif    v8.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v9.16b, %[vzero].16b, v19.16b             \n" \
  "fmla   v11.8h, v2.8h, %[wr11].8h                 \n" \
  "fmla   v12.8h, v3.8h, %[wr12].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr10].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v5.16b, #14        \n" \
  "fmul   v13.8h, v4.8h, %[wr01].8h                 \n" \
  "fmla   v11.8h, v4.8h, %[wr21].8h                 \n" \
  "fmul   v14.8h, v5.8h, %[wr02].8h                 \n" \
  "fmla   v12.8h, v5.8h, %[wr22].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr00].8h                \n" \
  "fmla   v16.8h, v10.8h, %[wr20].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v7.16b, #14        \n" \
  "sub    %[din_ptr4], %[din_ptr4], #2              \n" \
  "fmla   v13.8h, v6.8h, %[wr11].8h                 \n" \
  "fmla   v14.8h, v7.8h, %[wr12].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr10].8h                \n" \
  "ext    v10.16b, %[vzero].16b, v9.16b, #14        \n" \
  "fadd   v16.8h, v16.8h, v11.8h                    \n" \
  "fadd   v16.8h, v16.8h, v12.8h                    \n" \
  "fmla   v13.8h, v8.8h, %[wr21].8h                 \n" \
  "fmla   v14.8h, v9.8h, %[wr22].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr20].8h                \n" \
  "fadd   v17.8h, v17.8h, v13.8h                    \n" \
  "fadd   v17.8h, v17.8h, v14.8h                    \n"

#define RIGHT_COMPUTE_FP16_S2P0_SMALL                   \
  "ldr    q18, [%[vmask]]                           \n" \
  "ldr    q19, [%[vmask], #0x10]                    \n" \
  "ldr    q20, [%[rmask]]                           \n" \
  "bif    v0.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v1.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v2.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v3.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v4.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v5.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v6.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v7.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v8.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v9.16b, %[vzero].16b, v19.16b             \n" \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ext    v10.16b, v0.16b, %[vzero].16b, #2         \n" \
  "fmul   v11.8h, v0.8h, %[wr00].8h                 \n" \
  "fmul   v12.8h, v1.8h, %[wr01].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr02].8h                \n" \
  "ext    v10.16b, v2.16b, %[vzero].16b, #2         \n" \
  "fmla   v11.8h, v2.8h, %[wr10].8h                 \n" \
  "fmla   v12.8h, v3.8h, %[wr11].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr12].8h                \n" \
  "ext    v10.16b, v4.16b, %[vzero].16b, #2         \n" \
  "fmul   v13.8h, v4.8h, %[wr00].8h                 \n" \
  "fmla   v11.8h, v4.8h, %[wr20].8h                 \n" \
  "fmul   v14.8h, v5.8h, %[wr01].8h                 \n" \
  "fmla   v12.8h, v5.8h, %[wr21].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr02].8h                \n" \
  "fmla   v16.8h, v10.8h, %[wr22].8h                \n" \
  "ext    v10.16b, v6.16b, %[vzero].16b, #2         \n" \
  "fmla   v13.8h, v6.8h, %[wr10].8h                 \n" \
  "fmla   v14.8h, v7.8h, %[wr11].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr12].8h                \n" \
  "ext    v10.16b, v8.16b, %[vzero].16b, #2         \n" \
  "fadd   v16.8h, v16.8h, v11.8h                    \n" \
  "fadd   v16.8h, v16.8h, v12.8h                    \n" \
  "fmla   v13.8h, v8.8h, %[wr20].8h                 \n" \
  "ld1    {v0.8h}, [%[ptr_out0]]                    \n" \
  "fmla   v14.8h, v9.8h, %[wr21].8h                 \n" \
  "ld1    {v1.8h}, [%[ptr_out1]]                    \n" \
  "fmla   v17.8h, v10.8h, %[wr22].8h                \n" \
  "bif    v16.16b, v0.16b, v20.16b                  \n" \
  "fadd   v17.8h, v17.8h, v13.8h                    \n" \
  "bif    v17.16b, v1.16b, v20.16b                  \n" \
  "fadd   v17.8h, v17.8h, v14.8h                    \n"

#else
#define INIT_FP16_S2                                          \
  "PLD    [%[din_ptr0]]                                   \n" \
  "PLD    [%[din_ptr1]]                                   \n" \
  "PLD    [%[din_ptr2]]                                   \n" \
  "PLD    [%[din_ptr3]]                                   \n" \
  "PLD    [%[din_ptr4]]                                   \n" \
  "PLD    [%[weight_ptr]]                                 \n" \
  "vld2.16    {d4-d7}, [%[din_ptr0]]!     @ q2 q3         \n" \
  "vld2.16    {d8-d11}, [%[din_ptr1]]!    @ q4 q5         \n" \
  "vld2.16    {d12-d15}, [%[din_ptr2]]!   @ q6 q7         \n" \
  "vld2.16    {d16-d19}, [%[din_ptr3]]!   @ q8 q9         \n" \
  "vld2.16    {d20-d23}, [%[din_ptr4]]!   @ q10 q11       \n" \
  "vld1.16    {d0, d1}, [%[weight_ptr]]   @ wr0-8 q0      \n" \
  "add       %[weight_ptr], %[weight_ptr], #16            \n"

#define LEFT_COMPUTE_FP16_S2                                  \
  "vld1.16    {d28, d29}, [%[bias_val]]      @ out_14     \n" \
  "veor       q13, q13, q13                  @ zero_13    \n" \
  "veor       q12, q12, q12                               \n" \
  "veor       q15, q15, q15                               \n" \
  "vext.8     q1, q13, q3, #14                            \n" \
  "vmla.f16   q12, q2, d0[1]                              \n" \
  "vmla.f16   q15, q3, d0[2]                              \n" \
  "vmla.f16   q14, q1, d0[0]                              \n" \
  "vext.8     q1, q13, q5, #14                            \n" \
  "sub    %[din_ptr0], %[din_ptr0], #2                    \n" \
  "sub    %[din_ptr1], %[din_ptr1], #2                    \n" \
  "vmla.f16   q12, q4, d1[0]                              \n" \
  "vmla.f16   q15, q5, d1[1]                              \n" \
  "vmla.f16   q14, q1, d0[3]                              \n" \
  "vld1.16    {d6[3]}, [%[weight_ptr]]       @ wr9 in q3  \n" \
  "vext.8    q1, q13, q7, #14                             \n" \
  "sub    %[din_ptr2], %[din_ptr2], #2                    \n" \
  "sub    %[din_ptr3], %[din_ptr3], #2                    \n" \
  "vld1.16    {d10, d11}, [%[bias_val]]                   \n" \
  "veor       q4, q4, q4                                  \n" \
  "veor       q2, q2, q2                                  \n" \
  "vmla.f16   q2, q6 , d0[1]                              \n" \
  "vmla.f16   q4, q7 , d0[2]                              \n" \
  "vmla.f16   q5, q1, d0[0]                               \n" \
  "vmla.f16   q12, q6, d1[3]                              \n" \
  "vmla.f16   q15, q7, d6[3]                              \n" \
  "vmla.f16   q14, q1, d1[2]                              \n" \
  "vext.8     q1, q13, q9, #14                            \n" \
  "sub    %[din_ptr4], %[din_ptr4], #2                    \n" \
  "vmla.f16   q2, q8 , d1[0]                              \n" \
  "vadd.f16   q14, q14, q12                               \n" \
  "vmla.f16   q4, q9 , d1[1]                              \n" \
  "vadd.f16   q14, q14, q15             @ q14 for out l1  \n" \
  "vmla.f16   q5, q1, d0[3]                               \n" \
  "vext.8    q1, q13, q11, #14                            \n" \
  "vmla.f16   q2, q10 , d1[3]                             \n" \
  "vmla.f16   q4, q11 , d6[3]                             \n" \
  "vmla.f16   q5, q1, d1[2]                               \n" \
  "vadd.f16   q15, q2, q4                                 \n" \
  "vadd.f16   q15, q15, q5              @ q15 for out l2  \n" \
  "vld2.16    {d4-d7}, [%[din_ptr0]]!     @ q2 q3         \n" \
  "vld2.16    {d8-d11}, [%[din_ptr1]]!    @ q4 q5         \n" \
  "vld2.16    {d12-d15}, [%[din_ptr2]]!   @ q6 q7         \n" \
  "vld2.16    {d16-d19}, [%[din_ptr3]]!   @ q8 q9         \n" \
  "vld2.16    {d20-d23}, [%[din_ptr4]]!   @ q10 q11       \n"

#define LEFT_RESULT_FP16_S2                                  \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "cmp    %[cnt], #1                                     \n" \
  "blt    1f                                             \n"

#define LEFT_RESULT_FP16_S2_RELU                             \
  "veor       q12, q12, q12                              \n" \
  "vmax.f16   q14,  q14, q12                             \n" \
  "vmax.f16   q15,  q15, q12                             \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "cmp    %[cnt], #1                                     \n" \
  "blt    1f                                             \n"

#define LEFT_RESULT_FP16_S2_RELU6                            \
  "vld1.16    {d24, d25}, [%[six_ptr]]    @ q12          \n" \
  "veor       q13,  q13, q13                             \n" \
  "vmax.f16   q14,  q14, q13                             \n" \
  "vmax.f16   q15,  q15, q13                             \n" \
  "vmin.f16   q14,  q14, q12                             \n" \
  "vmin.f16   q15,  q15, q12                             \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "cmp    %[cnt], #1                                     \n" \
  "blt    1f                                             \n"

#define LEFT_RESULT_FP16_S2_LEAKY_RELU                       \
  "vld1.16    {d26, d27}, [%[scale_ptr]]      @ q13      \n" \
  "veor       q12, q12, q12                              \n" \
  "vcge.f16   q1, q14, q12                               \n" \
  "vmul.f16   q12, q14, q13                              \n" \
  "vbif.8     q14, q12, q1                               \n" \
  "veor       q12, q12, q12                              \n" \
  "vcge.f16   q1, q15, q12                               \n" \
  "vmul.f16   q12, q15, q13                              \n" \
  "vbif.8     q15, q12, q1                               \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "cmp    %[cnt], #1                                     \n" \
  "blt    1f                                             \n"

#define MID_COMPUTE_FP16_S2                                  \
  "2:                                                    \n" \
  "vld1.16    {d24, d25}, [%[din_ptr0]]   @ tmp: q12     \n" \
  "vld1.16    {d28, d29}, [%[bias_val]]   @ out1: q14    \n" \
  "veor       q13, q13, q13                              \n" \
  "veor       q15, q15, q15                              \n" \
  "vext.8     q1, q2, q12, #2                            \n" \
  "vmla.f16   q13, q2, d0[0]                             \n" \
  "vmla.f16   q15, q3, d0[1]                             \n" \
  "vmla.f16   q14, q1, d0[2]                             \n" \
  "vld1.16    {d24, d25}, [%[din_ptr1]]   @ tmp: q12     \n" \
  "vext.8     q1, q4, q12, #2                            \n" \
  "vmla.f16   q13, q4, d0[3]                             \n" \
  "vmla.f16   q15, q5, d1[0]                             \n" \
  "vmla.f16   q14, q1, d1[1]                             \n" \
  "vld1.16    {d6[3]}, [%[weight_ptr]]    @ wr9 in q3    \n" \
  "vld1.16    {d24, d25}, [%[din_ptr2]]   @ tmp: q12     \n" \
  "veor       q2, q2, q2                                 \n" \
  "vext.8     q1, q6, q12, #2                            \n" \
  "vmla.f16   q13, q6, d1[2]                             \n" \
  "vmla.f16   q15, q7, d1[3]                             \n" \
  "vmla.f16   q14, q1, d6[3]                             \n" \
  "vadd.f16   q14, q14, q15                              \n" \
  "vadd.f16   q14, q14, q13            @ out1: q14       \n" \
  "veor       q5, q5, q5                                 \n" \
  "vld1.16    {d30, d31}, [%[bias_val]]                  \n" \
  "vmla.f16   q2, q6, d0[0]                              \n" \
  "vmla.f16   q5, q7, d0[1]                              \n" \
  "vmla.f16   q15, q1, d0[2]                             \n" \
  "vld1.16    {d24, d25}, [%[din_ptr3]]   @ tmp: q12     \n" \
  "vext.8     q1, q8, q12, #2                            \n" \
  "vmla.f16   q2, q8, d0[3]                              \n" \
  "vmla.f16   q5, q9, d1[0]                              \n" \
  "vmla.f16   q15, q1, d1[1]                             \n" \
  "vld1.16    {d24, d25}, [%[din_ptr4]]   @ tmp: q12     \n" \
  "vext.8     q1, q10, q12, #2                           \n" \
  "vmla.f16   q2, q10, d1[2]                             \n" \
  "vmla.f16   q5, q11, d1[3]                             \n" \
  "vmla.f16   q15, q1, d6[3]                             \n" \
  "vadd.f16   q15, q15, q2                               \n" \
  "vadd.f16   q15, q15, q5             @ out2: q15       \n" \
  "vld2.16    {d4-d7}, [%[din_ptr0]]!     @ q2 q3        \n" \
  "vld2.16    {d8-d11}, [%[din_ptr1]]!    @ q4 q5        \n" \
  "vld2.16    {d12-d15}, [%[din_ptr2]]!   @ q6 q7        \n" \
  "vld2.16    {d16-d19}, [%[din_ptr3]]!   @ q8 q9        \n" \
  "vld2.16    {d20-d23}, [%[din_ptr4]]!   @ q10 q11      \n"

#define MID_RESULT_FP16_S2                                   \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "subs       %[cnt], %[cnt], #1                         \n" \
  "bne    2b                                             \n" \
  "1:                                                    \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16           \n"

#define MID_RESULT_FP16_S2_RELU                              \
  "veor       q13, q13, q13                              \n" \
  "vmax.f16   q14, q14, q13                              \n" \
  "vmax.f16   q15, q15, q13                              \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "subs       %[cnt], %[cnt], #1                         \n" \
  "bne    2b                                             \n" \
  "1:                                                    \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16           \n"

#define MID_RESULT_FP16_S2_RELU6                             \
  "vld1.16    {d24, d25}, [%[six_ptr]]                   \n" \
  "veor       q13, q13, q13                              \n" \
  "vmax.f16   q14, q14, q13                              \n" \
  "vmax.f16   q15, q15, q13                              \n" \
  "vmin.f16   q14, q14, q12                              \n" \
  "vmin.f16   q15, q15, q12                              \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "subs       %[cnt], %[cnt], #1                         \n" \
  "bne    2b                                             \n" \
  "1:                                                    \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16           \n"

#define MID_RESULT_FP16_S2_LEAKY_RELU                        \
  "vld1.16    {d26, d27}, [%[scale_ptr]]      @ q13      \n" \
  "veor       q12, q12, q12                              \n" \
  "vcge.f16   q1, q14, q12                               \n" \
  "vmul.f16   q12, q14, q13                              \n" \
  "vbif.8     q14, q12, q1                               \n" \
  "veor       q12, q12, q12                              \n" \
  "vcge.f16   q1, q15, q12                               \n" \
  "vmul.f16   q12, q15, q13                              \n" \
  "vbif.8     q15, q12, q1                               \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]!                 \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]!                 \n" \
  "subs       %[cnt], %[cnt], #1                         \n" \
  "bne    2b                                             \n" \
  "1:                                                    \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16           \n"

#define RIGHT_COMPUTE_FP16_S2                                 \
  "cmp    %[remain], #1                                   \n" \
  "blt    4f                                              \n" \
  "3:                                                     \n" \
  "vld1.16    {d0, d1}, [%[weight_ptr]]   @ wr0-8 q0      \n" \
  "vld1.16    {d24-d27}, [%[vmask]]!  @ mask: q12,q13,q1  \n" \
  "vld1.16    {d2, d3}, [%[vmask]]                        \n" \
  "vld1.16    {d28, d29}, [%[bias_val]]      @ out1: q14  \n" \
  "add       %[weight_ptr], %[weight_ptr], #16            \n" \
  "sub    %[ptr_out0], %[ptr_out0], %[right_st_num]       \n" \
  "sub    %[ptr_out1], %[ptr_out1], %[right_st_num]       \n" \
  "lsl    %[right_st_num], #1                             \n" \
  "add    %[right_st_num], #32                            \n" \
  "sub    %[din_ptr0], %[din_ptr0], %[right_st_num]       \n" \
  "sub    %[din_ptr1], %[din_ptr1], %[right_st_num]       \n" \
  "sub    %[din_ptr2], %[din_ptr2], %[right_st_num]       \n" \
  "sub    %[din_ptr3], %[din_ptr3], %[right_st_num]       \n" \
  "sub    %[din_ptr4], %[din_ptr4], %[right_st_num]       \n" \
  "sub    %[right_st_num], #32                            \n" \
  "lsr    %[right_st_num], #1                             \n" \
  "vld2.16    {d4-d7}, [%[din_ptr0]]       @ q2 q3        \n" \
  "vld2.16    {d8-d11}, [%[din_ptr1]]      @ q4 q5        \n" \
  "vld2.16    {d12-d15}, [%[din_ptr2]]     @ q6 q7        \n" \
  "veor      q15, q15, q15                 @ zero q15     \n" \
  "vbif.8    q2, q15, q12                                 \n" \
  "vbif.8    q3, q15, q13                                 \n" \
  "vbif.8    q4, q15, q12                                 \n" \
  "vbif.8    q5, q15, q13                                 \n" \
  "vbif.8    q6, q15, q12                                 \n" \
  "vbif.8    q7, q15, q13                                 \n" \
  "add    %[din_ptr0], %[din_ptr0], #4                    \n" \
  "add    %[din_ptr1], %[din_ptr1], #4                    \n" \
  "add    %[din_ptr2], %[din_ptr2], #4                    \n" \
  "vld2.16    {d16-d19}, [%[din_ptr0]]    @ tmp: q8 q9    \n" \
  "vld2.16    {d20-d23}, [%[din_ptr1]]    @ tmp: q10 q11  \n" \
  "vbif.8     q8, q15, q1                                 \n" \
  "vmla.f16   q14, q2, d0[0]                              \n" \
  "vmla.f16   q14, q3, d0[1]                              \n" \
  "vmla.f16   q14, q8, d0[2]                              \n" \
  "vbif.8     q10, q15, q1                                \n" \
  "vmla.f16   q14, q4, d0[3]                              \n" \
  "vmla.f16   q14, q5, d1[0]                              \n" \
  "vmla.f16   q14, q10, d1[1]                             \n" \
  "vld2.16    {d4-d7}, [%[din_ptr2]]    @ tmp: q2 q3      \n" \
  "vld1.16    {d6[3]}, [%[weight_ptr]]    @ wr9 in q3     \n" \
  "vbif.8     q2, q15, q1                                 \n" \
  "vmla.f16   q14, q6, d1[2]                              \n" \
  "vmla.f16   q14, q7, d1[3]                              \n" \
  "vmla.f16   q14, q2, d6[3]             @ out1: q14      \n" \
  "veor      q4, q4, q4                                   \n" \
  "veor      q5, q5, q5                                   \n" \
  "vld1.16    {d30, d31}, [%[bias_val]]    @ out2: q15    \n" \
  "vld2.16    {d16-d19}, [%[din_ptr3]]     @ q8 q9        \n" \
  "vld2.16    {d20-d23}, [%[din_ptr4]]     @ q10 q11      \n" \
  "vmla.f16   q4, q6, d0[0]                               \n" \
  "vmla.f16   q15, q7, d0[1]                              \n" \
  "vmla.f16   q15, q2, d0[2]                              \n" \
  "add    %[din_ptr3], %[din_ptr3], #4                    \n" \
  "add    %[din_ptr4], %[din_ptr4], #4                    \n" \
  "vbif.8    q8, q5, q12                                  \n" \
  "vbif.8    q9, q5, q13                                  \n" \
  "vbif.8    q10, q5, q12                                 \n" \
  "vbif.8    q11, q5, q13                                 \n" \
  "vld2.16    {d12-d15}, [%[din_ptr3]]    @ tmp: q6 q7    \n" \
  "vbif.8    q6, q5, q1                                   \n" \
  "vmla.f16   q4, q8, d0[3]                               \n" \
  "vmla.f16   q15, q9, d1[0]                              \n" \
  "vmla.f16   q15, q6, d1[1]                              \n" \
  "vld2.16    {d12-d15}, [%[din_ptr4]]    @ tmp: q6 q7    \n" \
  "vbif.8    q6, q5, q1                                   \n" \
  "vmla.f16   q4, q10, d1[2]                              \n" \
  "vmla.f16   q15, q11, d1[3]                             \n" \
  "vmla.f16   q15, q6, d6[3]                              \n" \
  "vadd.f16   q15, q15, q4                                \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16            \n"

#define RIGHT_RESULT_FP16_S2                                 \
  "vst1.16    {d28, d29}, [%[ptr_out0]]                  \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]                  \n" \
  "4:                                                    \n"

#define RIGHT_RESULT_FP16_S2_RELU                            \
  "veor       q13, q13, q13                              \n" \
  "vmax.f16   q14, q14, q13                              \n" \
  "vmax.f16   q15, q15, q13                              \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]                  \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]                  \n" \
  "4:                                                    \n"

#define RIGHT_RESULT_FP16_S2_RELU6                           \
  "vld1.16    {d24, d25}, [%[six_ptr]]                   \n" \
  "veor       q13, q13, q13                              \n" \
  "vmax.f16   q14, q14, q13                              \n" \
  "vmax.f16   q15, q15, q13                              \n" \
  "vmin.f16   q14, q14, q12                              \n" \
  "vmin.f16   q15, q15, q12                              \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]                  \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]                  \n" \
  "4:                                                    \n"

#define RIGHT_RESULT_FP16_S2_LEAKY_RELU                      \
  "vld1.16    {d26, d27}, [%[scale_ptr]]      @ q13      \n" \
  "veor       q12, q12, q12                              \n" \
  "vcge.f16   q1, q14, q12                               \n" \
  "vmul.f16   q12, q14, q13                              \n" \
  "vbif.8     q14, q12, q1                               \n" \
  "veor       q12, q12, q12                              \n" \
  "vcge.f16   q1, q15, q12                               \n" \
  "vmul.f16   q12, q15, q13                              \n" \
  "vbif.8     q15, q12, q1                               \n" \
  "vst1.16    {d28, d29}, [%[ptr_out0]]                  \n" \
  "vst1.16    {d30, d31}, [%[ptr_out1]]                  \n" \
  "4:                                                    \n"

#define RIGHT_COMPUTE_FP16_S2P1_SMALL                         \
  "vld1.16    {d24-d27}, [%[vmask]]!  @ mask: q12,q13     \n" \
  "vld1.16    {d28, d29}, [%[bias_val]]      @ out1: q14  \n" \
  "veor      q15, q15, q15                 @ zero q15     \n" \
  "vbif.8    q2, q15, q12                                 \n" \
  "vbif.8    q3, q15, q13                                 \n" \
  "vbif.8    q4, q15, q12                                 \n" \
  "vbif.8    q5, q15, q13                                 \n" \
  "vbif.8    q6, q15, q12                                 \n" \
  "vbif.8    q7, q15, q13                                 \n" \
  "vext.8    q1, q15, q3, #14                             \n" \
  "vmla.f16   q14, q2, d0[1]                              \n" \
  "vmla.f16   q14, q3, d0[2]                              \n" \
  "vmla.f16   q14, q1, d0[0]                              \n" \
  "vext.8     q1, q15, q5, #14                            \n" \
  "vmla.f16   q14, q4, d1[0]                              \n" \
  "vmla.f16   q14, q5, d1[1]                              \n" \
  "vmla.f16   q14, q1, d0[3]                              \n" \
  "vld1.16    {d6[3]}, [%[weight_ptr]]    @ wr9 in q3     \n" \
  "vext.8     q1, q15, q7, #14                            \n" \
  "vbif.8    q8, q15, q12                                 \n" \
  "vbif.8    q9, q15, q13                                 \n" \
  "vbif.8    q10, q15, q12                                \n" \
  "vbif.8    q11, q15, q13                                \n" \
  "vmla.f16   q14, q6, d1[3]                              \n" \
  "vmla.f16   q14, q7, d6[3]                              \n" \
  "vmla.f16   q14, q1, d1[2]             @ out1: q14      \n" \
  "vld1.16    {d4, d5}, [%[bias_val]]    @ tmp: q2        \n" \
  "vmla.f16   q2, q6, d0[1]                               \n" \
  "vmla.f16   q2, q7, d0[2]                               \n" \
  "vmla.f16   q2, q1, d0[0]                               \n" \
  "vext.8     q1, q15, q9, #14                            \n" \
  "vmla.f16   q2, q8, d1[0]                               \n" \
  "vmla.f16   q2, q9, d1[1]                               \n" \
  "vmla.f16   q2, q1, d0[3]                               \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16            \n" \
  "vext.8     q1, q15, q11, #14                           \n" \
  "vmla.f16   q2, q10, d1[3]                              \n" \
  "vmla.f16   q2, q11, d6[3]                              \n" \
  "vmla.f16   q15, q1, d1[2]                              \n" \
  "vadd.f16   q15, q15, q2                                \n" \
  "sub    %[din_ptr4], %[din_ptr4], #2                    \n"

#define RIGHT_COMPUTE_FP16_S2P0_SMALL                         \
  "vld1.16    {d24-d27}, [%[vmask]]!  @ mask: q12,q13     \n" \
  "veor      q15, q15, q15                 @ zero q15     \n" \
  "vbif.8    q2, q15, q12                                 \n" \
  "vbif.8    q3, q15, q13                                 \n" \
  "vbif.8    q4, q15, q12                                 \n" \
  "vbif.8    q5, q15, q13                                 \n" \
  "vbif.8    q6, q15, q12                                 \n" \
  "vbif.8    q7, q15, q13                                 \n" \
  "vbif.8    q8, q15, q12                                 \n" \
  "vbif.8    q9, q15, q13                                 \n" \
  "vbif.8    q10, q15, q12                                \n" \
  "vbif.8    q11, q15, q13                                \n" \
  "vld1.16    {d28, d29}, [%[bias_val]]    @ q14          \n" \
  "vext.8     q1, q2, q15, #2                             \n" \
  "vmla.f16   q14, q2, d0[0]                              \n" \
  "vmla.f16   q14, q3, d0[1]                              \n" \
  "vmla.f16   q14, q1, d0[2]                              \n" \
  "vext.8    q1, q4, q15, #2                              \n" \
  "vld1.16    {d6[3]}, [%[weight_ptr]]    @ wr9 in q3     \n" \
  "vmla.f16   q14, q4, d0[3]                              \n" \
  "vmla.f16   q14, q5, d1[0]                              \n" \
  "vmla.f16   q14, q1, d1[1]                              \n" \
  "vext.8    q1, q6, q15, #2                              \n" \
  "vmla.f16   q14, q6, d1[2]                              \n" \
  "vmla.f16   q14, q7, d1[3]                              \n" \
  "vmla.f16   q14, q1, d6[3]                              \n" \
  "vld1.16    {d4, d5}, [%[bias_val]]    @ q2             \n" \
  "vmla.f16   q2, q6, d0[0]                               \n" \
  "vmla.f16   q2, q7, d0[1]                               \n" \
  "vmla.f16   q2, q1, d0[2]                               \n" \
  "vext.8    q1, q8, q15, #2                              \n" \
  "vmla.f16   q2, q8, d0[3]                               \n" \
  "vmla.f16   q2, q9, d1[0]                               \n" \
  "vmla.f16   q2, q1, d1[1]                               \n" \
  "vext.8    q1, q10, q15, #2                             \n" \
  "sub       %[weight_ptr], %[weight_ptr], #16            \n" \
  "vmla.f16   q2, q10, d1[2]                              \n" \
  "vmla.f16   q2, q11, d1[3]                              \n" \
  "vmla.f16   q15, q1, d6[3]                              \n" \
  "vadd.f16   q15, q15, q2                                \n" \
  "vld1.16    {d6, d7}, [%[rmask]]!                       \n" \
  "vld1.16    {d2, d3}, [%[ptr_out0]]                     \n" \
  "vld1.16    {d4, d5}, [%[ptr_out1]]                     \n" \
  "vbif.8    q14, q1, q3                                  \n" \
  "vbif.8    q15, q2, q3                                  \n"

#endif

#define FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, alpha) \
  float16x8_t wr00 = vdupq_n_f16(weight_ptr[0]);            \
  float16x8_t wr10 = vdupq_n_f16(weight_ptr[3]);            \
  float16x8_t wr20 = vdupq_n_f16(weight_ptr[6]);            \
  float16x8_t wr01 = vdupq_n_f16(weight_ptr[1]);            \
  float16x8_t wr11 = vdupq_n_f16(weight_ptr[4]);            \
  float16x8_t wr21 = vdupq_n_f16(weight_ptr[7]);            \
  float16x8_t wr02 = vdupq_n_f16(weight_ptr[2]);            \
  float16x8_t wr12 = vdupq_n_f16(weight_ptr[5]);            \
  float16x8_t wr22 = vdupq_n_f16(weight_ptr[8]);            \
  float16x8_t vzero = vdupq_n_f16(0.f);                     \
  float16_t v_bias[16] = {0.f};                             \
  for (int i = 0; i < 8; i++) {                             \
    v_bias[i] = bias_val;                                   \
    v_bias[i + 8] = alpha;                                  \
  }

#define INIT_PTR_3x3_S2_FP16(din, w_in) \
  float16_t* doutr0 = nullptr;          \
  float16_t* doutr1 = nullptr;          \
  const float16_t* dr0 = din;           \
  const float16_t* dr1 = dr0 + w_in;    \
  const float16_t* dr2 = dr1 + w_in;    \
  const float16_t* dr3 = dr2 + w_in;    \
  const float16_t* dr4 = dr3 + w_in;    \
  const float16_t* din_ptr0 = nullptr;  \
  const float16_t* din_ptr1 = nullptr;  \
  const float16_t* din_ptr2 = nullptr;  \
  const float16_t* din_ptr3 = nullptr;  \
  const float16_t* din_ptr4 = nullptr;

#define ASSIGN_PTR_3x3_S2_FP16(w_out) \
  din_ptr0 = dr0;                     \
  din_ptr1 = dr1;                     \
  din_ptr2 = dr2;                     \
  din_ptr3 = dr3;                     \
  din_ptr4 = dr4;                     \
  doutr0 = dout_ptr;                  \
  doutr1 = doutr0 + w_out;

#define TOP_BOTTOM_BORDER_3x3_S2P1_FP16(w_in, h_in, h_out) \
  if (i == 0) {                                            \
    din_ptr0 = zero_ptr;                                   \
    din_ptr1 = dr0;                                        \
    din_ptr2 = dr1;                                        \
    din_ptr3 = dr2;                                        \
    din_ptr4 = dr3;                                        \
    dr0 = dr3;                                             \
    dr1 = dr4;                                             \
  } else {                                                 \
    dr0 = dr4;                                             \
    dr1 = dr0 + w_in;                                      \
  }                                                        \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
  dr4 = dr3 + w_in;                                        \
  if (i + 4 > h_in) {                                      \
    switch (i + 4 - h_in) {                                \
      case 4:                                              \
        din_ptr1 = zero_ptr;                               \
      case 3:                                              \
        din_ptr2 = zero_ptr;                               \
      case 2:                                              \
        din_ptr3 = zero_ptr;                               \
      case 1:                                              \
        din_ptr4 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i / 2 + 2 > h_out) {                                 \
    doutr1 = write_ptr;                                    \
  }

#define TOP_BOTTOM_BORDER_3x3_S2P0_FP16(w_in, h_in, h_out) \
  dr0 = dr4;                                               \
  dr1 = dr0 + w_in;                                        \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
  dr4 = dr3 + w_in;                                        \
  if (i * 2 + 5 > h_in) {                                  \
    switch (i * 2 + 5 - h_in) {                            \
      case 4:                                              \
        din_ptr1 = zero_ptr;                               \
      case 3:                                              \
        din_ptr2 = zero_ptr;                               \
      case 2:                                              \
        din_ptr3 = zero_ptr;                               \
      case 1:                                              \
        din_ptr4 = zero_ptr;                               \
      case 0:                                              \
        din_ptr4 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 2 > h_out) {                                     \
    doutr1 = write_ptr;                                    \
  }
#define SMALL_TMP_ADDR          \
  float16_t tmp_out[2][8];      \
  float16_t* tmp0 = tmp_out[0]; \
  float16_t* tmp1 = tmp_out[1];
#define SMALL_REAL_STORE            \
  for (int j = 0; j < w_out; j++) { \
    *(doutr0 + j) = tmp0[j];        \
    *(doutr1 + j) = tmp1[j];        \
  }

inline std::pair<uint16_t, uint16_t> right_mask_3x3s2p01_fp16(int w_in,
                                                              int w_out,
                                                              int pad,
                                                              uint16_t* vmask) {
  const uint16_t right_pad_idx[24] = {0, 2, 4, 6, 8,  10, 12, 14,
                                      1, 3, 5, 7, 9,  11, 13, 15,
                                      2, 4, 6, 8, 10, 12, 14, 16};
  const uint16_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int tile_w = w_out >> 3;

  int cnt_remain = w_out % 8;
  bool no_right_compute = ((w_out % 8) == 0) && ((2 * w_out + 1 - pad <= w_in));
  uint16_t size_right_remain = w_in - (2 * w_out + 1 - pad - 17);
  cnt_remain = ((!no_right_compute && (w_out % 8) == 0) ? 8 : cnt_remain);
  tile_w = tile_w - (cnt_remain == 8 ? 1 : 0) - pad;

  uint16x8_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx));
  uint16x8_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 8));
  uint16x8_t vmask_rp3 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 16));

  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);
  vst1q_u16(vmask + 16, vmask_rp3);

  return std::make_pair(tile_w, cnt_remain);
}

inline void right_mask_3x3_s2_small_fp16(int w_in,
                                         int w_out,
                                         uint16_t* vmask,
                                         uint16_t* rmask) {
  const uint16_t right_pad_idx[16] = {
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
  const uint16_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16_t size_right_remain = w_in;
  uint16_t cnt_remain = w_out;
  uint16x8_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx));
  uint16x8_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 8));
  uint16x8_t rmask_rp =
      vcgtq_u16(vdupq_n_u16(cnt_remain), vld1q_u16(out_pad_idx));
  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);
  vst1q_u16(rmask, rmask_rp);
}

// w_in > 16
void conv_depthwise_3x3s2p1_bias_noact_common_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* scale,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 1, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0, j = 0;
                                                  i < h_in, j < h_out;
                                                  i += 4, j += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 LEFT_COMPUTE_FP16_S2 LEFT_RESULT_FP16_S2
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), [vmask] "r" (val_mask), \
              [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21");
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          LEFT_COMPUTE_FP16_S2 
          LEFT_RESULT_FP16_S2
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), \
            [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"); 
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 16
void conv_depthwise_3x3s2p1_bias_relu_common_fp16_fp16(float16_t* dout,
                                                       const float16_t* din,
                                                       const float16_t* weights,
                                                       const float16_t* bias,
                                                       const float16_t* scale,
                                                       bool flag_bias,
                                                       int num,
                                                       int ch_in,
                                                       int h_in,
                                                       int w_in,
                                                       int h_out,
                                                       int w_out,
                                                       ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 1, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0, j = 0;
                                                  i < h_in, j < h_out;
                                                  i += 4, j += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 LEFT_COMPUTE_FP16_S2 LEFT_RESULT_FP16_S2_RELU
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2_RELU
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2_RELU
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [vmask] "r" (val_mask),
              [bias_val] "r"(v_bias),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          LEFT_COMPUTE_FP16_S2 
          LEFT_RESULT_FP16_S2_RELU
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2_RELU
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2_RELU
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), \
            [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s2p1_bias_relu6_common_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* six,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 1, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, six[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0, j = 0;
                                                  i < h_in, j < h_out;
                                                  i += 4, j += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 LEFT_COMPUTE_FP16_S2 LEFT_RESULT_FP16_S2_RELU6
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2_RELU6
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2_RELU6
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [vmask] "r" (val_mask),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          LEFT_COMPUTE_FP16_S2 
          LEFT_RESULT_FP16_S2_RELU6
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2_RELU6
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [six_ptr] "r"(six)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2_RELU6
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), [six_ptr] "r"(six),\
            [right_st_num] "r"(right_st_num)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s2p1_bias_leaky_relu_common_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* scale,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 1, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, scale[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0, j = 0;
                                                  i < h_in, j < h_out;
                                                  i += 4, j += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 LEFT_COMPUTE_FP16_S2 LEFT_RESULT_FP16_S2_LEAKY_RELU
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2_LEAKY_RELU
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2_LEAKY_RELU
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [vmask] "r" (val_mask),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)              
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          LEFT_COMPUTE_FP16_S2 
          LEFT_RESULT_FP16_S2_LEAKY_RELU
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2_LEAKY_RELU
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [scale_ptr] "r"(scale)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2_LEAKY_RELU
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), [scale_ptr] "r"(scale),\
            [right_st_num] "r"(right_st_num)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 16
void conv_depthwise_3x3s2p0_bias_noact_common_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* scale,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 0, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2          
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [vmask] "r" (val_mask),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)              
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), [right_st_num] "r"(right_st_num)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 16
void conv_depthwise_3x3s2p0_bias_relu_common_fp16_fp16(float16_t* dout,
                                                       const float16_t* din,
                                                       const float16_t* weights,
                                                       const float16_t* bias,
                                                       const float16_t* scale,
                                                       bool flag_bias,
                                                       int num,
                                                       int ch_in,
                                                       int h_in,
                                                       int w_in,
                                                       int h_out,
                                                       int w_out,
                                                       ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 0, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2_RELU
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2_RELU
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vmask] "r" (val_mask),
              [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)              
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2_RELU
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2_RELU
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), [right_st_num] "r"(right_st_num)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s2p0_bias_relu6_common_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* six,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 0, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, six[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2_RELU6
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2_RELU6
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [vmask] "r" (val_mask),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2_RELU6
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [six_ptr] "r"(six)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2_RELU6
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), [right_st_num] "r"(right_st_num), [six_ptr] "r"(six)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_3x3s2p0_bias_leaky_relu_common_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* scale,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[24];
  auto&& res = right_mask_3x3s2p01_fp16(w_in, w_out, 0, vmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  uint16_t right_pad_num = (8 - cnt_remain) * 4 + 32;
  uint16_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, scale[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2_LEAKY_RELU
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2_LEAKY_RELU
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [vmask] "r" (val_mask),
              [right_pad_num] "r"(right_pad_num), 
              [right_st_num] "r"(right_st_num)              
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        // general registers used in inline asm must be less than 13,
        // dividing them into two part.
        asm volatile(
          INIT_FP16_S2 
          MID_COMPUTE_FP16_S2 
          MID_RESULT_FP16_S2_LEAKY_RELU
          : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [scale_ptr] "r"(scale)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        asm volatile(
          RIGHT_COMPUTE_FP16_S2 
          RIGHT_RESULT_FP16_S2_LEAKY_RELU
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
            [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
            [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
            [weight_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(v_bias), [remain] "r"(cnt_remain), [right_st_num] "r"(right_st_num), [scale_ptr] "r"(scale)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p1_bias_noact_small_fp16_fp16(float16_t* dout,
                                                       const float16_t* din,
                                                       const float16_t* weights,
                                                       const float16_t* bias,
                                                       const float16_t* scale,
                                                       bool flag_bias,
                                                       int num,
                                                       int ch_in,
                                                       int h_in,
                                                       int w_in,
                                                       int h_out,
                                                       int w_out,
                                                       ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);

  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(
          din_ch_ptr, w_in) for (int i = 0; i < h_in && i / 2 < h_out; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask)
            : [bias_val] "r"(v_bias), [weight_ptr] "r"(weight_ptr)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p1_bias_relu_small_fp16_fp16(float16_t* dout,
                                                      const float16_t* din,
                                                      const float16_t* weights,
                                                      const float16_t* bias,
                                                      const float16_t* scale,
                                                      bool flag_bias,
                                                      int num,
                                                      int ch_in,
                                                      int h_in,
                                                      int w_in,
                                                      int h_out,
                                                      int w_out,
                                                      ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(
          din_ch_ptr, w_in) for (int i = 0; i < h_in && i / 2 < h_out; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask)
            : [weight_ptr] "r"(weight_ptr),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p1_bias_relu6_small_fp16_fp16(float16_t* dout,
                                                       const float16_t* din,
                                                       const float16_t* weights,
                                                       const float16_t* bias,
                                                       const float16_t* six,
                                                       bool flag_bias,
                                                       int num,
                                                       int ch_in,
                                                       int h_in,
                                                       int w_in,
                                                       int h_out,
                                                       int w_out,
                                                       ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, six[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(
          din_ch_ptr, w_in) for (int i = 0; i < h_in && i / 2 < h_out; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2_RELU6
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2_RELU6
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask)
            : [weight_ptr] "r"(weight_ptr),
              [bias_val] "r"(v_bias),
              [six_ptr] "r"(six)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p1_bias_leaky_relu_small_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* scale,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, scale[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(
          din_ch_ptr, w_in) for (int i = 0; i < h_in && i / 2 < h_out; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2_LEAKY_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P1_SMALL RIGHT_RESULT_FP16_S2_LEAKY_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask)
            : [weight_ptr] "r"(weight_ptr),
              [bias_val] "r"(v_bias),
              [scale_ptr] "r"(scale)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 16
void conv_depthwise_3x3s2p0_bias_noact_small_fp16_fp16(float16_t* dout,
                                                       const float16_t* din,
                                                       const float16_t* weights,
                                                       const float16_t* bias,
                                                       const float16_t* scale,
                                                       bool flag_bias,
                                                       int num,
                                                       int ch_in,
                                                       int h_in,
                                                       int w_in,
                                                       int h_out,
                                                       int w_out,
                                                       ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [weight_ptr] "r"(weight_ptr),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p0_bias_relu_small_fp16_fp16(float16_t* dout,
                                                      const float16_t* din,
                                                      const float16_t* weights,
                                                      const float16_t* bias,
                                                      const float16_t* scale,
                                                      bool flag_bias,
                                                      int num,
                                                      int ch_in,
                                                      int h_in,
                                                      int w_in,
                                                      int h_out,
                                                      int w_out,
                                                      ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, 0.f)
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
      asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [weight_ptr] "r"(weight_ptr),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p0_bias_relu6_small_fp16_fp16(float16_t* dout,
                                                       const float16_t* din,
                                                       const float16_t* weights,
                                                       const float16_t* bias,
                                                       const float16_t* six,
                                                       bool flag_bias,
                                                       int num,
                                                       int ch_in,
                                                       int h_in,
                                                       int w_in,
                                                       int h_out,
                                                       int w_out,
                                                       ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, six[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2_RELU6
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2_RELU6
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [weight_ptr] "r" (weight_ptr),
              [bias_val] "r"(v_bias),
              [six_ptr] "r"(six)
           : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 16
void conv_depthwise_3x3s2p0_bias_leaky_relu_small_fp16_fp16(
    float16_t* dout,
    const float16_t* din,
    const float16_t* weights,
    const float16_t* bias,
    const float16_t* scale,
    bool flag_bias,
    int num,
    int ch_in,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    ARMContext* ctx) {
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  uint16_t vmask[16];
  uint16_t rmask[8];
  right_mask_3x3_s2_small_fp16(w_in, w_out, vmask, rmask);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      SMALL_TMP_ADDR
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
#ifdef __aarch64__
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val, scale[0])
#else
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#endif
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2_LEAKY_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
        asm volatile(
          INIT_FP16_S2 RIGHT_COMPUTE_FP16_S2P0_SMALL RIGHT_RESULT_FP16_S2_LEAKY_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(tmp0),
              [ptr_out1] "+r"(tmp1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [weight_ptr] "r"(weight_ptr),
              [bias_val] "r"(v_bias),
              [scale_ptr] "r"(scale)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",\
            "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += 2 * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
