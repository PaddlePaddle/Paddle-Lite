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
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

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
  "ld1    {v21.8h}, [%[six_ptr]]                    \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    1f                                        \n"

#define LEFT_RESULT_FP16_S2_LEAKY_RELU                  \
  "ld1    {v21.8h}, [%[scale_ptr]]                  \n" \
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
  "bne    2b                                        \n"

#define MID_RESULT_FP16_S2_RELU                         \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "bne    2b                                        \n"

#define MID_RESULT_FP16_S2_RELU6                        \
  "ld1    {v21.8h}, [%[six_ptr]]                    \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "bne    2b                                        \n"

#define MID_RESULT_FP16_S2_LEAKY_RELU                   \
  "ld1    {v21.8h}, [%[scale_ptr]]                  \n" \
  "fcmge  v12.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v12.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "bne    2b                                        \n"

#define RIGHT_COMPUTE_FP16_S2                           \
  "1:                                               \n" \
  "cmp    %w[remain], #1                            \n" \
  "blt    4f                                        \n" \
  "3:                                               \n" \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ld1    {v18.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v19.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v20.8h}, [%[rmask]], #16                 \n" \
  "bif    v0.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v1.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v2.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v3.16b, %[vzero].16b, v19.16b             \n" \
  "bif    v4.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v5.16b, %[vzero].16b, v19.16b             \n" \
  "ext    v10.16b, v0.16b, %[vzero].16b, #2         \n" \
  "bif    v6.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v7.16b, %[vzero].16b, v19.16b             \n" \
  "fmul   v11.8h, v0.8h, %[wr00].8h                 \n" \
  "fmul   v12.8h, v1.8h, %[wr01].8h                 \n" \
  "fmla   v16.8h, v10.8h, %[wr02].8h                \n" \
  "ext    v10.16b, v2.16b, %[vzero].16b, #2         \n" \
  "bif    v8.16b, %[vzero].16b, v18.16b             \n" \
  "bif    v9.16b, %[vzero].16b, v19.16b             \n" \
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
  "ld1    {v0.8h}, [%[ptr_out0]]                    \n" \
  "fadd   v16.8h, v16.8h, v11.8h                    \n" \
  "fadd   v16.8h, v16.8h, v12.8h                    \n" \
  "ld1    {v1.8h}, [%[ptr_out1]]                    \n" \
  "fmla   v13.8h, v8.8h, %[wr20].8h                 \n" \
  "fmla   v14.8h, v9.8h, %[wr21].8h                 \n" \
  "fmla   v17.8h, v10.8h, %[wr22].8h                \n" \
  "bif    v16.16b, v0.16b, v20.16b                  \n" \
  "fadd   v17.8h, v17.8h, v13.8h                    \n" \
  "fadd   v17.8h, v17.8h, v14.8h                    \n" \
  "bif    v17.16b, v1.16b, v20.16b                  \n"

#define RIGHT_RESULT_FP16_S2_RELU                       \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "4:                                               \n"

#define RIGHT_RESULT_FP16_S2                            \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "4:                                               \n"

#define RIGHT_RESULT_FP16_S2_RELU6                      \
  "ld1    {v21.8h}, [%[six_ptr]]                    \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "4:                                               \n"

#define RIGHT_RESULT_FP16_S2_LEAKY_RELU                 \
  "ld1    {v21.8h}, [%[scale_ptr]]                  \n" \
  "fcmge  v12.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v12.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "4:                                               \n"

#define RIGHT_COMPUTE_FP16_S2P1_SMALL                   \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ld1    {v18.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v19.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v20.8h}, [%[rmask]], #16                 \n" \
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
  "ld1    {v0.8h}, [%[ptr_out0]]                    \n" \
  "fmla   v14.8h, v9.8h, %[wr22].8h                 \n" \
  "bif    v16.16b, v0.16b, v20.16b                  \n" \
  "ld1    {v1.8h}, [%[ptr_out1]]                    \n" \
  "fmla   v17.8h, v10.8h, %[wr20].8h                \n" \
  "fadd   v17.8h, v17.8h, v13.8h                    \n" \
  "fadd   v17.8h, v17.8h, v14.8h                    \n" \
  "bif    v17.16b, v1.16b, v20.16b                  \n"

#define RIGHT_COMPUTE_FP16_S2P0_SMALL                   \
  "ld1    {v18.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v19.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v20.8h}, [%[rmask]], #16                 \n" \
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
#endif

#define FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val) \
  float16x8_t wr00 = vdupq_n_f16(weight_ptr[0]);     \
  float16x8_t wr10 = vdupq_n_f16(weight_ptr[3]);     \
  float16x8_t wr20 = vdupq_n_f16(weight_ptr[6]);     \
  float16x8_t wr01 = vdupq_n_f16(weight_ptr[1]);     \
  float16x8_t wr11 = vdupq_n_f16(weight_ptr[4]);     \
  float16x8_t wr21 = vdupq_n_f16(weight_ptr[7]);     \
  float16x8_t wr02 = vdupq_n_f16(weight_ptr[2]);     \
  float16x8_t wr12 = vdupq_n_f16(weight_ptr[5]);     \
  float16x8_t wr22 = vdupq_n_f16(weight_ptr[8]);     \
  float16x8_t vzero = vdupq_n_f16(0.f);              \
  float16_t v_bias[8] = {bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val};

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

inline std::pair<uint16_t, uint16_t> right_mask_3x3s2p1_fp16(int w_in,
                                                             int w_out,
                                                             uint16_t* vmask,
                                                             uint16_t* rmask) {
  const uint16_t right_pad_idx[16] = {
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
  const uint16_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16_t cnt_col = ((w_out >> 3) - 2);
  uint16_t size_right_remain =
      static_cast<uint16_t>(w_in - (15 + cnt_col * 16));
  if (size_right_remain >= 17) {
    cnt_col++;
    size_right_remain -= 16;
  }
  uint16_t cnt_remain = (size_right_remain == 16 && w_out % 8 == 0)
                            ? 8
                            : static_cast<uint16_t>(w_out % 8);
  uint16x8_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx));
  uint16x8_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 8));
  uint16x8_t rmask_rp =
      vcgtq_u16(vdupq_n_u16(cnt_remain), vld1q_u16(out_pad_idx));
  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);
  vst1q_u16(rmask, rmask_rp);
  return std::make_pair(cnt_col, cnt_remain);
}

inline std::pair<uint16_t, uint16_t> right_mask_3x3s2p0_fp16(int w_in,
                                                             int w_out,
                                                             uint16_t* vmask,
                                                             uint16_t* rmask) {
  const uint16_t right_pad_idx[16] = {
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
  const uint16_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int tile_w = w_out >> 3;
  int cnt_remain = w_out % 8;
  uint16_t size_right_remain = (uint16_t)(16 + (tile_w << 4) - w_in);
  size_right_remain = 16 - size_right_remain;
  if (cnt_remain == 0 && size_right_remain == 0) {
    cnt_remain = 8;
    tile_w -= 1;
    size_right_remain = 16;
  }
  uint16x8_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx));
  uint16x8_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 8));
  uint16x8_t rmask_rp =
      vcgtq_u16(vdupq_n_u16(cnt_remain), vld1q_u16(out_pad_idx));
  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);
  vst1q_u16(rmask, rmask_rp);
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p1_fp16(w_in, w_out, vmask, rmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S2 LEFT_COMPUTE_FP16_S2 LEFT_RESULT_FP16_S2
          MID_COMPUTE_FP16_S2 MID_RESULT_FP16_S2
          RIGHT_COMPUTE_FP16_S2 RIGHT_RESULT_FP16_S2
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (val_mask), \
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), [remain] "r"(cnt_remain)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p1_fp16(w_in, w_out, vmask, rmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p1_fp16(w_in, w_out, vmask, rmask);
  int cnt_col = res.first;
  int cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain),
              [six_ptr] "r"(six)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p1_fp16(w_in, w_out, vmask, rmask);
  int cnt_col = res.first;
  int cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P1_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain),
              [scale_ptr] "r"(scale)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p0_fp16(w_in, w_out, vmask, rmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p0_fp16(w_in, w_out, vmask, rmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p0_fp16(w_in, w_out, vmask, rmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain),
              [six_ptr] "r"(six)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
  uint16_t vmask[16];
  uint16_t rmask[8];
  auto&& res = right_mask_3x3s2p0_fp16(w_in, w_out, vmask, rmask);
  uint16_t cnt_col = res.first;
  uint16_t cnt_remain = res.second;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += 2) {
        ASSIGN_PTR_3x3_S2_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S2P0_FP16(
            w_in, h_in, h_out) int cnt = cnt_col;
        uint16_t* rst_mask = rmask;
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
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain),
              [scale_ptr] "r"(scale)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [six_ptr] "r"(six)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_in; i += 4) {
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [scale_ptr] "r"(scale)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [six_ptr] "r"(six)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
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
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
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
              [bias_val] "r"(v_bias),
              [scale_ptr] "r"(scale)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
