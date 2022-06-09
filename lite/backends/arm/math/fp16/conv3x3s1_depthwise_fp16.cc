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
#define INIT_FP16_S1                                    \
  "PRFM PLDL1KEEP, [%[din_ptr0]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr1]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr2]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr3]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr4]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr5]]                    \n" \
  "ld1    {v0.8h}, [%[din_ptr0]], #16               \n" \
  "ld1    {v2.8h}, [%[din_ptr1]], #16               \n" \
  "ld1    {v4.8h}, [%[din_ptr2]], #16               \n" \
  "ld1    {v6.8h}, [%[din_ptr3]], #16               \n" \
  "ld1    {v1.8h}, [%[din_ptr0]]                    \n" \
  "ld1    {v3.8h}, [%[din_ptr1]]                    \n" \
  "ld1    {v5.8h}, [%[din_ptr2]]                    \n" \
  "ld1    {v7.8h}, [%[din_ptr3]]                    \n" \
  "ld1    {v8.8h}, [%[din_ptr4]], #16               \n" \
  "ld1    {v10.8h}, [%[din_ptr5]], #16              \n" \
  "ld1    {v9.8h}, [%[din_ptr4]]                    \n" \
  "ld1    {v11.8h}, [%[din_ptr5]]                   \n" \
  "mov v12.16b, %[ww].16b                           \n"
#define LEFT_COMPUTE_FP16_S1                                     \
  "ld1    {v16.8h}, [%[bias_val]]                   \n"          \
  "ld1    {v17.8h}, [%[bias_val]]                   \n"          \
  "ld1    {v18.8h}, [%[bias_val]]                   \n"          \
  "ld1    {v19.8h}, [%[bias_val]]                   \n"          \
  "ext    v22.16b, %[vzero].16b, v0.16b, #14        \n"          \
  "ext    v13.16b, v0.16b, v1.16b, #2               \n"          \
  "fmla   v16.8h,  v0.8h, v12.h[1]                \n" /* r0 */   \
  "sub    %[din_ptr0], %[din_ptr0], #2              \n"          \
  "sub    %[din_ptr1], %[din_ptr1], #2              \n"          \
  "fmla   v16.8h, v22.8h ,v12.h[0]                 \n"           \
  "fmla   v16.8h, v13.8h ,v12.h[2]                \n"            \
  "ext    v14.16b, %[vzero].16b, v2.16b, #14        \n"          \
  "ext    v15.16b, v2.16b, v3.16b, #2               \n"          \
  "ext    v22.16b, %[vzero].16b, v4.16b, #14        \n" /* r1 */ \
  "fmla   v17.8h, v2.8h, v12.h[1]                  \n"           \
  "fmla   v16.8h, v2.8h, v12.h[4]                  \n"           \
  "ext    v13.16b, v4.16b, v5.16b, #2               \n"          \
  "fmla   v17.8h, v14.8h, v12.h[0]                 \n"           \
  "fmla   v16.8h, v14.8h, v12.h[3]                 \n"           \
  "fmla   v17.8h, v15.8h, v12.h[2]                 \n"           \
  "fmla   v16.8h, v15.8h, v12.h[5]                 \n"           \
  "sub    %[din_ptr2], %[din_ptr2], #2              \n"          \
  "sub    %[din_ptr3], %[din_ptr3], #2              \n"          \
  "ext    v14.16b, %[vzero].16b, v6.16b, #14        \n" /* r2 */ \
  "ld1    {v0.8h}, [%[din_ptr0]], #16               \n"          \
  "fmla   v18.8h, v4.8h, v12.h[1]                  \n"           \
  "fmla   v17.8h, v4.8h, v12.h[4]                  \n"           \
  "fmla   v16.8h, v4.8h, v12.h[7]                  \n"           \
  "ext    v15.16b, v6.16b, v7.16b, #2               \n"          \
  "ld1    {v2.8h}, [%[din_ptr1]], #16               \n"          \
  "fmla   v18.8h, v22.8h, v12.h[0]                 \n"           \
  "fmla   v17.8h, v22.8h, v12.h[3]                 \n"           \
  "fmla   v16.8h, v22.8h, v12.h[6]                 \n"           \
  "ld1    {v1.8h}, [%[din_ptr0]]                    \n"          \
  "fmla   v18.8h, v13.8h, v12.h[2]                 \n"           \
  "fmla   v17.8h, v13.8h, v12.h[5]                 \n"           \
  "fmla   v16.8h, %[ww8].8h, v13.8h                 \n"          \
  "ld1    {v3.8h}, [%[din_ptr1]]                    \n"          \
  "sub    %[din_ptr4], %[din_ptr4], #2              \n"          \
  "sub    %[din_ptr5], %[din_ptr5], #2              \n"          \
  "ext    v22.16b, %[vzero].16b, v8.16b, #14        \n" /* r3 */ \
  "ld1    {v4.8h}, [%[din_ptr2]], #16               \n"          \
  "fmla   v19.8h, v14.8h, v12.h[0]                 \n"           \
  "fmla   v18.8h, v14.8h, v12.h[3]                 \n"           \
  "fmla   v17.8h, v14.8h, v12.h[6]                 \n"           \
  "ext    v13.16b, v8.16b, v9.16b, #2               \n"          \
  "ld1    {v5.8h}, [%[din_ptr2]]                    \n"          \
  "fmla   v19.8h, v6.8h, v12.h[1]                  \n"           \
  "fmla   v18.8h, v6.8h, v12.h[4]                  \n"           \
  "fmla   v17.8h, v6.8h, v12.h[7]                  \n"           \
  "ld1    {v6.8h}, [%[din_ptr3]], #16               \n"          \
  "fmla   v19.8h, v15.8h, v12.h[2]                 \n"           \
  "fmla   v18.8h, v15.8h, v12.h[5]                 \n"           \
  "fmla   v17.8h, %[ww8].8h, v15.8h                 \n"          \
  "ld1    {v7.8h}, [%[din_ptr3]]                    \n"          \
  "ext    v14.16b, %[vzero].16b, v10.16b, #14       \n" /* r4 */ \
  "fmla   v19.8h, v22.8h, v12.h[3]                 \n"           \
  "fmla   v18.8h, v22.8h, v12.h[6]                 \n"           \
  "fmla   v19.8h, v8.8h, v12.h[4]                  \n"           \
  "fmla   v18.8h, v8.8h, v12.h[7]                  \n"           \
  "ext    v15.16b, v10.16b, v11.16b, #2             \n"          \
  "fmla   v19.8h, v13.8h, v12.h[5]                 \n"           \
  "fmla   v18.8h, %[ww8].8h, v13.8h                 \n"          \
  "fmla   v19.8h, v14.8h, v12.h[6]                 \n" /* r5 */  \
  "fmla   v19.8h, v10.8h, v12.h[7]                 \n"           \
  "fmla   v19.8h, %[ww8].8h, v15.8h                 \n"          \
  "ld1    {v8.8h}, [%[din_ptr4]], #16               \n"          \
  "ld1    {v9.8h}, [%[din_ptr4]]                    \n"          \
  "ld1    {v10.8h}, [%[din_ptr5]], #16              \n"          \
  "ld1    {v11.8h}, [%[din_ptr5]]                   \n"

#define LEFT_RESULT_FP16_S1_RELU                        \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmax   v18.8h,  v18.8h, %[vzero].8h              \n" \
  "fmax   v19.8h,  v19.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    3f                                        \n"

#define LEFT_RESULT_FP16_S1                             \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    3f                                        \n"

#define LEFT_RESULT_FP16_S1_RELU6                       \
  "ld1    {v20.8h}, [%[six_ptr]]                    \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmax   v18.8h,  v18.8h, %[vzero].8h              \n" \
  "fmax   v19.8h,  v19.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v20.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v20.8h                   \n" \
  "fmin   v18.8h,  v18.8h, v20.8h                   \n" \
  "fmin   v19.8h,  v19.8h, v20.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    3f                                        \n"

#define LEFT_RESULT_FP16_S1_LEAKY_RELU                  \
  "ld1    {v21.8h}, [%[scale_ptr]]                  \n" \
  "fcmge  v22.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v22.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "fcmge  v22.8h,  v18.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v18.8h,  v21.8h                  \n" \
  "bif    v18.16b, v13.16b, v22.16b                 \n" \
  "fcmge  v14.8h,  v19.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v19.8h,  v21.8h                 \n"  \
  "bif    v19.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "cmp    %w[cnt], #1                               \n" \
  "blt    3f                                        \n"

#define MID_COMPUTE_FP16_S1                                      \
  "1:                                               \n"          \
  "ld1    {v16.8h}, [%[bias_val]]                   \n"          \
  "ld1    {v17.8h}, [%[bias_val]]                   \n"          \
  "ld1    {v18.8h}, [%[bias_val]]                   \n"          \
  "ld1    {v19.8h}, [%[bias_val]]                   \n"          \
  "fmla   v16.8h, v0.8h, v12.h[0]                  \n" /* r0 */  \
  "ext    v22.16b, v0.16b, v1.16b, #2               \n"          \
  "ext    v13.16b, v0.16b, v1.16b, #4               \n"          \
  "fmla   v16.8h, v22.8h, v12.h[1]                 \n"           \
  "fmla   v16.8h, v13.8h, v12.h[2]                 \n"           \
  "ext    v14.16b, v2.16b, v3.16b, #2               \n"          \
  "ext    v15.16b, v2.16b, v3.16b, #4               \n"          \
  "ext    v22.16b, v4.16b, v5.16b, #2               \n" /* r1 */ \
  "fmla   v17.8h, v2.8h, v12.h[0]                  \n"           \
  "fmla   v16.8h, v2.8h, v12.h[3]                  \n"           \
  "ext    v13.16b, v4.16b, v5.16b, #4               \n"          \
  "fmla   v17.8h, v14.8h, v12.h[1]                 \n"           \
  "fmla   v16.8h, v14.8h, v12.h[4]                 \n"           \
  "fmla   v17.8h, v15.8h, v12.h[2]                 \n"           \
  "fmla   v16.8h, v15.8h, v12.h[5]                 \n"           \
  "ext    v14.16b, v6.16b, v7.16b, #2               \n" /* r2 */ \
  "ld1    {v0.8h}, [%[din_ptr0]], #16               \n"          \
  "fmla   v18.8h, v4.8h, v12.h[0]                  \n"           \
  "fmla   v17.8h, v4.8h, v12.h[3]                  \n"           \
  "fmla   v16.8h, v4.8h, v12.h[6]                  \n"           \
  "ext    v15.16b, v6.16b, v7.16b, #4               \n"          \
  "ld1    {v2.8h}, [%[din_ptr1]], #16               \n"          \
  "fmla   v18.8h, v22.8h, v12.h[1]                 \n"           \
  "fmla   v17.8h, v22.8h, v12.h[4]                 \n"           \
  "fmla   v16.8h, v22.8h, v12.h[7]                 \n"           \
  "ld1    {v1.8h}, [%[din_ptr0]]                    \n"          \
  "fmla   v18.8h, v13.8h, v12.h[2]                 \n"           \
  "fmla   v17.8h, v13.8h, v12.h[5]                 \n"           \
  "fmla   v16.8h, %[ww8].8h, v13.8h                 \n"          \
  "ld1    {v3.8h}, [%[din_ptr1]]                    \n" /* r3 */ \
  "ext    v22.16b, v8.16b, v9.16b, #2               \n"          \
  "ld1    {v4.8h}, [%[din_ptr2]], #16               \n"          \
  "fmla   v19.8h, v6.8h, v12.h[0]                  \n"           \
  "fmla   v18.8h, v6.8h, v12.h[3]                  \n"           \
  "fmla   v17.8h, v6.8h, v12.h[6]                  \n"           \
  "ext    v13.16b, v8.16b, v9.16b, #4               \n"          \
  "fmla   v19.8h, v14.8h, v12.h[1]                 \n"           \
  "fmla   v18.8h, v14.8h, v12.h[4]                 \n"           \
  "fmla   v17.8h, v14.8h, v12.h[7]                 \n"           \
  "ld1    {v5.8h}, [%[din_ptr2]]                    \n"          \
  "fmla   v19.8h, v15.8h, v12.h[2]                 \n"           \
  "fmla   v18.8h, v15.8h, v12.h[5]                 \n"           \
  "fmla   v17.8h, %[ww8].8h, v15.8h                 \n"          \
  "ext    v14.16b, v10.16b, v11.16b, #2             \n" /* r4 */ \
  "ld1    {v6.8h}, [%[din_ptr3]], #16               \n"          \
  "fmla   v19.8h, v8.8h, v12.h[3]                  \n"           \
  "fmla   v18.8h, v8.8h, v12.h[6]                  \n"           \
  "fmla   v19.8h, v22.8h, v12.h[4]                 \n"           \
  "ld1    {v7.8h}, [%[din_ptr3]]                    \n"          \
  "fmla   v18.8h, v22.8h, v12.h[7]                 \n"           \
  "ext    v15.16b, v10.16b, v11.16b, #4             \n"          \
  "fmla   v19.8h, v13.8h, v12.h[5]                 \n"           \
  "fmla   v18.8h, %[ww8].8h, v13.8h                 \n"          \
  "ld1    {v8.8h}, [%[din_ptr4]], #16               \n" /* r5 */ \
  "fmla   v19.8h, v10.8h, v12.h[6]                 \n"           \
  "ld1    {v9.8h}, [%[din_ptr4]]                    \n"          \
  "fmla   v19.8h, v14.8h, v12.h[7]                 \n"           \
  "ld1    {v10.8h}, [%[din_ptr5]], #16              \n"          \
  "fmla   v19.8h, %[ww8].8h, v15.8h                 \n"          \
  "ld1    {v11.8h}, [%[din_ptr5]]                   \n"          \
  "subs   %w[cnt], %w[cnt], #1                      \n"

#define MID_RESULT_FP16_S1_RELU                         \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmax   v18.8h,  v18.8h, %[vzero].8h              \n" \
  "fmax   v19.8h,  v19.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "bne    1b                                        \n"

#define MID_RESULT_FP16_S1                              \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "bne    1b                                        \n"

#define MID_RESULT_FP16_S1_RELU6                        \
  "ld1    {v21.8h}, [%[six_ptr]]                    \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmax   v18.8h,  v18.8h, %[vzero].8h              \n" \
  "fmax   v19.8h,  v19.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "fmin   v18.8h,  v18.8h, v21.8h                   \n" \
  "fmin   v19.8h,  v19.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "bne    1b                                        \n"

#define MID_RESULT_FP16_S1_LEAKY_RELU                   \
  "ld1    {v20.8h}, [%[scale_ptr]]                  \n" \
  "fcmge  v22.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v20.8h                  \n" \
  "bif    v16.16b, v13.16b, v22.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v20.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "fcmge  v22.8h,  v18.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v18.8h,  v20.8h                  \n" \
  "bif    v18.16b, v13.16b, v22.16b                 \n" \
  "fcmge  v14.8h,  v19.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v19.8h,  v20.8h                  \n" \
  "bif    v19.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]], #16              \n" \
  "st1    {v17.8h}, [%[ptr_out1]], #16              \n" \
  "st1    {v18.8h}, [%[ptr_out2]], #16              \n" \
  "st1    {v19.8h}, [%[ptr_out3]], #16              \n" \
  "bne    1b                                        \n"

#define RIGHT_COMPUTE_FP16_S1                                          \
  "3:                                                     \n"          \
  "ld1    {v20.8h, v21.8h}, [%[vmask]]                    \n"          \
  "ld1    {v16.8h}, [%[bias_val]]                         \n"          \
  "ld1    {v17.8h}, [%[bias_val]]                         \n"          \
  "ld1    {v18.8h}, [%[bias_val]]                         \n"          \
  "ld1    {v19.8h}, [%[bias_val]]                         \n"          \
  "sub    %[din_ptr0], %[din_ptr0], %[right_pad_num]      \n"          \
  "sub    %[din_ptr1], %[din_ptr1], %[right_pad_num]      \n"          \
  "sub    %[din_ptr2], %[din_ptr2], %[right_pad_num]      \n"          \
  "sub    %[din_ptr3], %[din_ptr3], %[right_pad_num]      \n"          \
  "sub    %[din_ptr4], %[din_ptr4], %[right_pad_num]      \n"          \
  "sub    %[din_ptr5], %[din_ptr5], %[right_pad_num]      \n"          \
  "ld1    {v0.8h, v1.8h}, [%[din_ptr0]]                   \n"          \
  "ld1    {v2.8h, v3.8h}, [%[din_ptr1]]                   \n"          \
  "sub    %[ptr_out0], %[ptr_out0], %[right_st_num]       \n"          \
  "sub    %[ptr_out1], %[ptr_out1], %[right_st_num]       \n"          \
  "sub    %[ptr_out2], %[ptr_out2], %[right_st_num]       \n"          \
  "sub    %[ptr_out3], %[ptr_out3], %[right_st_num]       \n"          \
  "ld1    {v4.8h, v5.8h}, [%[din_ptr2]]                   \n"          \
  "ld1    {v6.8h, v7.8h}, [%[din_ptr3]]                   \n"          \
  "bif    v0.16b, %[vzero].16b, v20.16b                   \n"          \
  "bif    v1.16b, %[vzero].16b, v21.16b                   \n"          \
  "bif    v2.16b, %[vzero].16b, v20.16b                   \n"          \
  "bif    v3.16b, %[vzero].16b, v21.16b                   \n"          \
  "ld1    {v8.8h, v9.8h}, [%[din_ptr4]]                   \n"          \
  "ld1    {v10.8h, v11.8h}, [%[din_ptr5]]                 \n"          \
  "ext    v22.16b, v0.16b, v1.16b, #2                     \n"          \
  "ext    v13.16b, v0.16b, v1.16b, #4                     \n"          \
  "fmla   v16.8h,  v0.8h,  v12.h[0]                      \n"           \
  "ext    v14.16b, v2.16b, v3.16b, #2                     \n" /*r0*/   \
  "ext    v15.16b, v2.16b, v3.16b, #4                     \n"          \
  "bif    v4.16b, %[vzero].16b, v20.16b                   \n"          \
  "bif    v5.16b, %[vzero].16b, v21.16b                   \n"          \
  "fmla   v16.8h,  v22.8h,  v12.h[1]                     \n"           \
  "bif    v6.16b, %[vzero].16b, v20.16b                   \n"          \
  "bif    v7.16b, %[vzero].16b, v21.16b                   \n"          \
  "fmla   v16.8h,  v13.8h,  v12.h[2]                     \n"           \
  "bif    v8.16b, %[vzero].16b, v20.16b                   \n" /*r1*/   \
  "bif    v9.16b, %[vzero].16b, v21.16b                   \n"          \
  "bif    v10.16b, %[vzero].16b, v20.16b                  \n"          \
  "bif    v11.16b, %[vzero].16b, v21.16b                  \n"          \
  "ext    v22.16b, v4.16b, v5.16b, #2                     \n"          \
  "fmla   v17.8h,  v2.8h,  v12.h[0]                      \n"           \
  "fmla   v16.8h,  v2.8h,  v12.h[3]                      \n"           \
  "ext    v13.16b, v4.16b, v5.16b, #4                     \n"          \
  "fmla   v17.8h,  v14.8h,  v12.h[1]                     \n"           \
  "fmla   v16.8h,  v14.8h,  v12.h[4]                     \n"           \
  "fmla   v17.8h,  v15.8h,  v12.h[2]                     \n"           \
  "fmla   v16.8h,  v15.8h,  v12.h[5]                     \n"           \
  "ext    v14.16b, v6.16b, v7.16b, #2                     \n" /*r2*/   \
  "fmla   v18.8h,  v4.8h,  v12.h[0]                      \n"           \
  "fmla   v17.8h,  v4.8h,  v12.h[3]                      \n"           \
  "fmla   v16.8h,  v4.8h,  v12.h[6]                      \n"           \
  "ext    v15.16b, v6.16b, v7.16b, #4                     \n"          \
  "fmla   v18.8h,  v22.8h,  v12.h[1]                     \n"           \
  "fmla   v17.8h,  v22.8h,  v12.h[4]                     \n"           \
  "fmla   v16.8h,  v22.8h,  v12.h[7]                     \n"           \
  "fmla   v18.8h,  v13.8h,  v12.h[2]                     \n"           \
  "fmla   v17.8h,  v13.8h,  v12.h[5]                     \n"           \
  "fmla   v16.8h,  %[ww8].8h,  v13.8h                     \n"          \
  "ext    v22.16b, v8.16b, v9.16b, #2                     \n" /* r3 */ \
  "fmla   v19.8h, v6.8h, v12.h[0]                        \n"           \
  "fmla   v18.8h, v6.8h, v12.h[3]                        \n"           \
  "fmla   v17.8h, v6.8h, v12.h[6]                        \n"           \
  "ext    v13.16b, v8.16b, v9.16b, #4                     \n"          \
  "fmla   v19.8h,  v14.8h,  v12.h[1]                     \n"           \
  "fmla   v18.8h,  v14.8h,  v12.h[4]                     \n"           \
  "fmla   v17.8h,  v14.8h,  v12.h[7]                     \n"           \
  "fmla   v19.8h, v15.8h, v12.h[2]                       \n"           \
  "fmla   v18.8h, v15.8h, v12.h[5]                       \n"           \
  "fmla   v17.8h, %[ww8].8h, v15.8h                       \n"          \
  "ext    v14.16b, v10.16b, v11.16b, #2                   \n" /* r4 */ \
  "fmla   v19.8h, v8.8h, v12.h[3]                        \n"           \
  "fmla   v18.8h, v8.8h, v12.h[6]                        \n"           \
  "fmla   v19.8h, v22.8h, v12.h[4]                       \n"           \
  "fmla   v18.8h, v22.8h, v12.h[7]                       \n"           \
  "ext    v15.16b, v10.16b, v11.16b, #4                   \n"          \
  "fmla   v19.8h, v13.8h, v12.h[5]                       \n"           \
  "fmla   v18.8h, %[ww8].8h, v13.8h                       \n"          \
  "fmla   v19.8h, v10.8h, v12.h[6]                       \n" /* r5 */  \
  "fmla   v19.8h, v14.8h, v12.h[7]                       \n"           \
  "fmla   v19.8h, %[ww8].8h, v15.8h                       \n"

#define RIGHT_RESULT_FP16_S1_RELU                       \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmax   v18.8h,  v18.8h, %[vzero].8h              \n" \
  "fmax   v19.8h,  v19.8h, %[vzero].8h              \n" \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "st1    {v18.8h}, [%[ptr_out2]]                   \n" \
  "st1    {v19.8h}, [%[ptr_out3]]                   \n"

#define RIGHT_RESULT_FP16_S1                            \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "st1    {v18.8h}, [%[ptr_out2]]                   \n" \
  "st1    {v19.8h}, [%[ptr_out3]]                   \n"

#define RIGHT_RESULT_FP16_S1_RELU6                      \
  "ld1    {v21.8h}, [%[six_ptr]]                    \n" \
  "fmax   v16.8h,  v16.8h, %[vzero].8h              \n" \
  "fmax   v17.8h,  v17.8h, %[vzero].8h              \n" \
  "fmax   v18.8h,  v18.8h, %[vzero].8h              \n" \
  "fmax   v19.8h,  v19.8h, %[vzero].8h              \n" \
  "fmin   v16.8h,  v16.8h, v21.8h                   \n" \
  "fmin   v17.8h,  v17.8h, v21.8h                   \n" \
  "fmin   v18.8h,  v18.8h, v21.8h                   \n" \
  "fmin   v19.8h,  v19.8h, v21.8h                   \n" \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "st1    {v18.8h}, [%[ptr_out2]]                   \n" \
  "st1    {v19.8h}, [%[ptr_out3]]                   \n"

#define RIGHT_RESULT_FP16_S1_LEAKY_RELU                 \
  "ld1    {v21.8h}, [%[scale_ptr]]                  \n" \
  "fcmge  v22.8h,  v16.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v16.8h,  v21.8h                  \n" \
  "bif    v16.16b, v13.16b, v22.16b                 \n" \
  "fcmge  v14.8h,  v17.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v17.8h,  v21.8h                  \n" \
  "bif    v17.16b, v15.16b, v14.16b                 \n" \
  "fcmge  v22.8h,  v18.8h,  %[vzero].8h             \n" \
  "fmul   v13.8h,  v18.8h,  v21.8h                  \n" \
  "bif    v18.16b, v13.16b, v22.16b                 \n" \
  "fcmge  v14.8h,  v19.8h,  %[vzero].8h             \n" \
  "fmul   v15.8h,  v19.8h,  v21.8h                  \n" \
  "bif    v19.16b, v15.16b, v14.16b                 \n" \
  "st1    {v16.8h}, [%[ptr_out0]]                   \n" \
  "st1    {v17.8h}, [%[ptr_out1]]                   \n" \
  "st1    {v18.8h}, [%[ptr_out2]]                   \n" \
  "st1    {v19.8h}, [%[ptr_out3]]                   \n"

#define INIT_FP16_S1_SMALL                              \
  "PRFM PLDL1KEEP, [%[din_ptr0]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr1]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr2]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr3]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr4]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr5]]                    \n" \
  "ld1    {v0.8h}, [%[din_ptr0]], #16               \n" \
  "ld1    {v2.8h}, [%[din_ptr1]], #16               \n" \
  "ld1    {v4.8h}, [%[din_ptr2]], #16               \n" \
  "ld1    {v6.8h}, [%[din_ptr3]], #16               \n" \
  "ld1    {v8.8h}, [%[din_ptr4]], #16               \n" \
  "ld1    {v10.8h}, [%[din_ptr5]], #16              \n" \
  "mov v12.16b, %[ww].16b                           \n"

#define RIGHT_COMPUTE_FP16_S1P1_SMALL                    \
  "ld1    {v20.8h}, [%[vmask]], #16                 \n"  \
  "ld1    {v21.8h}, [%[vmask]]                      \n"  \
  "bif    v0.16b, %[vzero].16b, v20.16b             \n"  \
  "bif    v2.16b, %[vzero].16b, v20.16b             \n"  \
  "bif    v4.16b, %[vzero].16b, v20.16b             \n"  \
  "bif    v6.16b, %[vzero].16b, v20.16b             \n"  \
  "bif    v8.16b, %[vzero].16b, v20.16b             \n"  \
  "bif    v10.16b, %[vzero].16b, v20.16b            \n"  \
  "ld1    {v1.8h}, [%[din_ptr0]]                    \n"  \
  "ld1    {v3.8h}, [%[din_ptr1]]                    \n"  \
  "ld1    {v5.8h}, [%[din_ptr2]]                    \n"  \
  "ld1    {v7.8h}, [%[din_ptr3]]                    \n"  \
  "ld1    {v9.8h}, [%[din_ptr4]]                    \n"  \
  "ld1    {v15.8h}, [%[din_ptr5]]                   \n"  \
  "bif    v1.16b, %[vzero].16b, v21.16b             \n"  \
  "bif    v3.16b, %[vzero].16b, v21.16b             \n"  \
  "bif    v5.16b, %[vzero].16b, v21.16b             \n"  \
  "bif    v7.16b, %[vzero].16b, v21.16b             \n"  \
  "bif    v9.16b, %[vzero].16b, v21.16b             \n"  \
  "bif    v15.16b, %[vzero].16b, v21.16b            \n"  \
  "ld1    {v16.8h}, [%[bias_val]]                   \n"  \
  "ld1    {v17.8h}, [%[bias_val]]                   \n"  \
  "ld1    {v18.8h}, [%[bias_val]]                   \n"  \
  "ld1    {v19.8h}, [%[bias_val]]                   \n"  \
  "fmla   v16.8h, v0.8h, v12.h[1]                  \n"   \
  "ext    v11.16b, %[vzero].16b, v0.16b, #14        \n"  \
  "ext    v22.16b, v0.16b, v1.16b, #2               \n"  \
  "fmla   v16.8h, v11.8h, v12.h[0]                 \n"   \
  "fmla   v16.8h, v22.8h, v12.h[2]                 \n"   \
  "ext    v13.16b, %[vzero].16b, v2.16b, #14        \n"  \
  "ext    v14.16b, v2.16b, v3.16b, #2               \n"  \
  "fmla   v17.8h, v2.8h, v12.h[1]                  \n"   \
  "fmla   v16.8h, v2.8h, v12.h[4]                  \n"   \
  "ext    v11.16b, %[vzero].16b, v4.16b, #14        \n"  \
  "ext    v22.16b, v4.16b, v5.16b, #2               \n"  \
  "fmla   v17.8h, v13.8h, v12.h[0]                 \n"   \
  "fmla   v16.8h, v13.8h, v12.h[3]                 \n"   \
  "fmla   v17.8h, v14.8h, v12.h[2]                 \n"   \
  "fmla   v16.8h, v14.8h, v12.h[5]                 \n"   \
  "fmla   v18.8h, v4.8h, v12.h[1]                  \n"   \
  "fmla   v17.8h, v4.8h, v12.h[4]                  \n"   \
  "fmla   v16.8h, v4.8h, v12.h[7]                  \n"   \
  "ext    v13.16b, %[vzero].16b, v6.16b, #14        \n"  \
  "ext    v14.16b, v6.16b, v7.16b, #2               \n"  \
  "fmla   v18.8h, v11.8h, v12.h[0]                 \n"   \
  "fmla   v17.8h, v11.8h, v12.h[3]                 \n"   \
  "fmla   v16.8h, v11.8h, v12.h[6]                 \n"   \
  "fmla   v18.8h, v22.8h, v12.h[2]                 \n"   \
  "fmla   v17.8h, v22.8h, v12.h[5]                 \n"   \
  "fmla   v16.8h, %[ww8].8h, v22.8h                 \n"  \
  "fmla   v19.8h, v6.8h, v12.h[1]                  \n"   \
  "fmla   v18.8h, v6.8h, v12.h[4]                  \n"   \
  "fmla   v17.8h, v6.8h, v12.h[7]                  \n"   \
  "ext    v11.16b, %[vzero].16b, v8.16b, #14        \n"  \
  "ext    v22.16b, v8.16b, v9.16b, #2               \n"  \
  "fmla   v19.8h, v13.8h, v12.h[0]                 \n"   \
  "fmla   v18.8h, v13.8h, v12.h[3]                 \n"   \
  "fmla   v17.8h, v13.8h, v12.h[6]                 \n"   \
  "fmla   v19.8h, v14.8h, v12.h[2]                 \n"   \
  "fmla   v18.8h, v14.8h, v12.h[5]                 \n"   \
  "fmla   v17.8h, %[ww8].8h, v14.8h                 \n"  \
  "fmla   v19.8h, v8.8h, v12.h[4]                  \n"   \
  "fmla   v18.8h, v8.8h, v12.h[7]                  \n"   \
  "ext    v13.16b, %[vzero].16b, v10.16b, #14       \n"  \
  "ext    v14.16b, v10.16b, V15.16b, #2             \n"  \
  "fmla   v19.8h, v11.8h, v12.h[3]                 \n"   \
  "fmla   v18.8h, v11.8h, v12.h[6]                 \n"   \
  "fmla   v19.8h, v22.8h, v12.h[5]                 \n"   \
  "fmla   v18.8h, %[ww8].8h, v22.8h                  \n" \
  "fmla   v19.8h, v10.8h, v12.h[7]                 \n"   \
  "fmla   v19.8h, v13.8h, v12.h[6]                 \n"   \
  "fmla   v19.8h, %[ww8].8h, v14.8h                 \n"

#define RIGHT_COMPUTE_FP16_S1P0_SMALL                   \
  "ld1    {v1.8h}, [%[din_ptr0]]                    \n" \
  "ld1    {v3.8h}, [%[din_ptr1]]                    \n" \
  "ld1    {v5.8h}, [%[din_ptr2]]                    \n" \
  "ld1    {v7.8h}, [%[din_ptr3]]                    \n" \
  "ld1    {v9.8h}, [%[din_ptr4]]                    \n" \
  "ld1    {v11.8h}, [%[din_ptr5]]                   \n" \
  "ld1    {v20.8h}, [%[vmask]], #16                 \n" \
  "ld1    {v21.8h}, [%[vmask]]                      \n" \
  "bif    v0.16b, %[vzero].16b, v20.16b             \n" \
  "bif    v2.16b, %[vzero].16b, v20.16b             \n" \
  "bif    v4.16b, %[vzero].16b, v20.16b             \n" \
  "bif    v6.16b, %[vzero].16b, v20.16b             \n" \
  "bif    v8.16b, %[vzero].16b, v20.16b             \n" \
  "bif    v10.16b, %[vzero].16b, v20.16b            \n" \
  "bif    v1.16b, %[vzero].16b, v21.16b             \n" \
  "bif    v3.16b, %[vzero].16b, v21.16b             \n" \
  "bif    v5.16b, %[vzero].16b, v21.16b             \n" \
  "bif    v7.16b, %[vzero].16b, v21.16b             \n" \
  "bif    v9.16b, %[vzero].16b, v21.16b             \n" \
  "bif    v11.16b, %[vzero].16b, v21.16b            \n" \
  "ld1    {v16.8h}, [%[bias_val]]                   \n" \
  "ld1    {v17.8h}, [%[bias_val]]                   \n" \
  "ld1    {v18.8h}, [%[bias_val]]                   \n" \
  "ld1    {v19.8h}, [%[bias_val]]                   \n" \
  "fmla   v16.8h, v0.8h, v12.h[0]                  \n"  \
  "ext    v22.16b, v0.16b, v1.16b, #2               \n" \
  "ext    v13.16b, v0.16b, v1.16b, #4               \n" \
  "fmla   v16.8h, v22.8h, v12.h[1]                 \n"  \
  "fmla   v16.8h, v13.8h, v12.h[2]                 \n"  \
  "ext    v14.16b, v2.16b, v3.16b, #2               \n" \
  "ext    v15.16b, v2.16b, v3.16b, #4               \n" \
  "fmla   v17.8h, v2.8h, v12.h[0]                  \n"  \
  "fmla   v16.8h, v2.8h, v12.h[3]                  \n"  \
  "ext    v22.16b, %[vzero].16b, v4.16b, #14        \n" \
  "ext    v13.16b, v4.16b, %[vzero].16b, #2         \n" \
  "fmla   v17.8h, v14.8h, v12.h[1]                 \n"  \
  "fmla   v16.8h, v14.8h, v12.h[3]                 \n"  \
  "fmla   v17.8h, v15.8h, v12.h[2]                 \n"  \
  "fmla   v16.8h, v15.8h, v12.h[5]                 \n"  \
  "fmla   v18.8h, v4.8h, v12.h[1]                  \n"  \
  "fmla   v17.8h, v4.8h, v12.h[4]                  \n"  \
  "fmla   v16.8h, v4.8h, v12.h[7]                  \n"  \
  "ext    v14.16b, %[vzero].16b, v6.16b, #14        \n" \
  "ext    v15.16b, v6.16b, %[vzero].16b, #2         \n" \
  "fmla   v18.8h, v22.8h, v12.h[0]                 \n"  \
  "fmla   v17.8h, v22.8h, v12.h[3]                 \n"  \
  "fmla   v16.8h, v22.8h, v12.h[6]                 \n"  \
  "fmla   v18.8h, v13.8h, v12.h[2]                 \n"  \
  "fmla   v17.8h, v13.8h, v12.h[5]                 \n"  \
  "fmla   v16.8h, %[ww8].8h, v13.8h                 \n" \
  "fmla   v19.8h, v6.8h, v12.h[1]                  \n"  \
  "fmla   v18.8h, v6.8h, v12.h[4]                  \n"  \
  "fmla   v17.8h, v6.8h, v12.h[7]                  \n"  \
  "ext    v22.16b, %[vzero].16b, v8.16b, #14        \n" \
  "ext    v13.16b, v8.16b, %[vzero].16b, #2         \n" \
  "fmla   v19.8h, v14.8h, v12.h[0]                 \n"  \
  "fmla   v18.8h, v14.8h, v12.h[3]                 \n"  \
  "fmla   v17.8h, v14.8h, v12.h[6]                 \n"  \
  "fmla   v19.8h, v15.8h, v12.h[2]                 \n"  \
  "fmla   v18.8h, v15.8h, v12.h[5]                 \n"  \
  "fmla   v17.8h, %[ww8].8h, v15.8h                 \n" \
  "fmla   v19.8h, v8.8h, v12.h[4]                  \n"  \
  "fmla   v18.8h, v8.8h, v12.h[7]                  \n"  \
  "ext    v14.16b, %[vzero].16b, v10.16b, #14       \n" \
  "ext    v15.16b, v10.16b, %[vzero].16b, #2        \n" \
  "fmla   v19.8h, v22.8h, v12.h[3]                 \n"  \
  "fmla   v18.8h, v22.8h, v12.h[6]                 \n"  \
  "fmla   v19.8h, v13.8h, v12.h[5]                 \n"  \
  "fmla   v18.8h, %[ww8].8h, v13.8h                 \n" \
  "fmla   v19.8h, v10.8h, v12.h[7]                 \n"  \
  "fmla   v19.8h, v14.8h, v12.h[6]                 \n"  \
  "fmla   v19.8h, %[ww8].8h, v15.8h                 \n" \
  "ld1    {v21.16b}, [%[rmask]], #16                \n" \
  "ld1    {v0.8h}, [%[ptr_out0]]                    \n" \
  "ld1    {v2.8h}, [%[ptr_out1]]                    \n" \
  "ld1    {v4.8h}, [%[ptr_out2]]                    \n" \
  "ld1    {v6.8h}, [%[ptr_out3]]                    \n" \
  "bif    v16.16b, v0.16b, v21.16b                  \n" \
  "bif    v17.16b, v2.16b, v21.16b                  \n" \
  "bif    v18.16b, v4.16b, v21.16b                  \n" \
  "bif    v19.16b, v6.16b, v21.16b                  \n"

#else
#define INIT_FP16_S1                         \
  "pld [%[din_ptr0]]   @ preload data   \n"  \
  "pld [%[din_ptr1]]                    \n"  \
  "pld [%[din_ptr2]]                    \n"  \
  "pld [%[din_ptr3]]                    \n"  \
  "vld1.16  {d8-d9}, [%[din_ptr0]]!    \n"   \
  "vld1.16  {d12-d13}, [%[din_ptr1]]!    \n" \
  "vld1.16  {d16-d17}, [%[din_ptr2]]!    \n" \
  "vld1.16  {d20-d21}, [%[din_ptr3]]!    \n" \
  "vld1.16  {d10-d11}, [%[din_ptr0]]    \n"  \
  "vld1.16  {d14-d15}, [%[din_ptr1]]   \n"   \
  "vld1.16  {d18-d19}, [%[din_ptr2]]    \n"  \
  "vld1.16  {d22-d23}, [%[din_ptr3]]    \n"

/* win <=8, only need to fetch q4, q6, q8, q10*/
/* no nedd fetch q5,q7,q9,q11 like INIT_FP16_S1  */
#define INIT_FP16_S1_SMALL                   \
  "pld [%[din_ptr0]]   @ preload data   \n"  \
  "pld [%[din_ptr1]]                    \n"  \
  "pld [%[din_ptr2]]                    \n"  \
  "pld [%[din_ptr3]]                    \n"  \
  "vld1.16  {d8-d9}, [%[din_ptr0]]!    \n"   \
  "vld1.16  {d12-d13}, [%[din_ptr1]]!    \n" \
  "vld1.16  {d16-d17}, [%[din_ptr2]]!    \n" \
  "vld1.16  {d20-d21}, [%[din_ptr3]]!    \n"

/* we have extract 4 rows into q4-5,q6-7,q8-9,q10-11！ */
/* q12 and q13 is to hold output */

#define LEFT_COMPUTE_FP16_S1                                                 \
  "vld1.16    {d24-d25}, [%[bias_val]]               \n"                     \
  "vld1.16    {d26-d27}, [%[bias_val]]                \n"                    \
  "vext.16    q14, %q[vzero], q4, #7     \n" /* row0 in：q14, 4, 15 */      \
  "vext.16    q15, q4, q5,  #1        \n"                                    \
  "vmla.f16   q12, q4, %e[ww][1]  \n"                                        \
  "vmla.f16   q12, q14, %e[ww][0]  \n"                                       \
  "vmla.f16   q12, q15, %e[ww][2]  \n"                                       \
  "vext.16    q14, %q[vzero], q6, #7     \n" /* row1 in q14，6，15 */      \
  "vext.16    q15, q6, q7,  #1        \n"                                    \
  "vmla.f16   q13, q6, %e[ww][1]  \n"                                        \
  "vmla.f16   q13, q14, %e[ww][0]  \n"                                       \
  "vmla.f16   q13, q15, %e[ww][2]  \n"                                       \
  "vmla.f16   q12, q6, %f[ww][0]  \n"                                        \
  "vmla.f16   q12, q14, %e[ww][3]  \n"                                       \
  "vmla.f16   q12, q15, %f[ww][1]  \n"                                       \
  "vext.16    q14, %q[vzero], q8, #7     \n" /* row3 in: q14，8，15 */     \
  "vext.16    q15, q8, q9, #1        \n"                                     \
  "vmla.f16   q13, q8, %f[ww][0]  \n"                                        \
  "vmla.f16   q13, q14, %e[ww][3]  \n"                                       \
  "vmla.f16   q13, q15, %f[ww][1]  \n"                                       \
  "vmla.f16   q12, q8, %f[ww][3]  \n"                                        \
  "vmla.f16   q12, q14, %f[ww][2]  \n"                                       \
  "vmla.f16   q12, q15, %q[ww8]  \n"                                         \
  "vext.16    q14, %q[vzero], q10, #7        \n" /* row3 in q14，10，15 */ \
  "vext.16    q15, q10, q11, #1        \n"                                   \
  "vmla.f16   q13, q10, %f[ww][3]  \n"                                       \
  "vmla.f16   q13, q14, %f[ww][2]  \n"                                       \
  "vmla.f16   q13, q15, %q[ww8]  \n"                                         \
  "sub    %[din_ptr0], %[din_ptr0], #2              \n"                      \
  "sub    %[din_ptr1], %[din_ptr1], #2              \n"                      \
  "sub    %[din_ptr2], %[din_ptr2], #2              \n"                      \
  "sub    %[din_ptr3], %[din_ptr3], #2              \n"                      \
  "vld1.16  {d8-d9},   [%[din_ptr0]]!    \n"                                 \
  "vld1.16  {d12-d13}, [%[din_ptr1]]!    \n"                                 \
  "vld1.16  {d16-d17}, [%[din_ptr2]]!    \n"                                 \
  "vld1.16  {d20-d21}, [%[din_ptr3]]!    \n"                                 \
  "vld1.16  {d10-d11}, [%[din_ptr0]]    \n"                                  \
  "vld1.16  {d14-d15}, [%[din_ptr1]]   \n"                                   \
  "vld1.16  {d18-d19}, [%[din_ptr2]]    \n"                                  \
  "vld1.16  {d22-d23}, [%[din_ptr3]]    \n"

/* q12 q13 is to hold output */
#define LEFT_RESULT_FP16_S1_RELU                         \
  "vmax.f16   q12,  q12, %q[vzero]              \n"      \
  "vmax.f16   q13,  q13, %q[vzero]              \n"      \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!            \n"   \
  "cmp    %[cnt], #1                               \n"   \
  "blt    3f                                        \n"

/* q12 q13 is to hold output */
#define LEFT_RESULT_FP16_S1                              \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!            \n"   \
  "cmp    %[cnt], #1                               \n"   \
  "blt    3f                                        \n"

/* q12 q13 is to hold output */
/* load six_ptr into q14 */
#define LEFT_RESULT_FP16_S1_RELU6                        \
  "vld1.16  {d28-d29}, [%[six_ptr]]    \n"               \
  "vmax.f16   q12,  q12, %q[vzero]              \n"      \
  "vmax.f16   q13,  q13, %q[vzero]              \n"      \
  "vmin.f16   q12,  q12, q14              \n"            \
  "vmin.f16   q13,  q13, q14              \n"            \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!            \n"   \
  "cmp    %[cnt], #1                               \n"   \
  "blt    3f                                        \n"

/* q12 , q13 is to hold output */
/* load scale_ptr into q14 */
#define LEFT_RESULT_FP16_S1_LEAKY_RELU                   \
  "vld1.16  {d28-d29}, [%[scale_ptr]]    \n"             \
  "vmul.f16 q15, q12, q14 \n"                            \
  "vcge.f16 q14, q12, %q[vzero]           \n"            \
  "vbif q12, q15, q14\n"                                 \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vld1.16  {d28-d29}, [%[scale_ptr]]    \n"             \
  "vmul.f16 q15, q13, q14 \n"                            \
  "vcge.f16 q14, q13, %q[vzero]           \n"            \
  "vbif q13, q15, q14\n"                                 \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!              \n" \
  "cmp    %[cnt], #1                               \n"   \
  "blt    3f                                        \n"

/* we have extract 4 rows into q4-5,q6-7,q8-9,q10-11！ */
/* q12 and q13 is to hold output */

#define MID_COMPUTE_FP16_S1                                                   \
  "1:                                               \n"                       \
  "vld1.16    {d24-d25}, [%[bias_val]]                   \n"                  \
  "vld1.16    {d26-d27}, [%[bias_val]]                   \n"                  \
  "vext.16    q14, q4,q5,  #1        \n" /* row0 input: q4, q14, q15 */       \
  "vext.16    q15, q4, q5,  #2        \n"                                     \
  "vmla.f16   q12, q4, %e[ww][0]  \n"                                         \
  "vmla.f16   q12, q14, %e[ww][1]  \n"                                        \
                                                                              \
  /* q4 is not used , so prefetch next q4 */                                  \
  "vld1.16  {d8-d9},   [%[din_ptr0]]!    \n"                                  \
  "vmla.f16   q12, q15, %e[ww][2]  \n"                                        \
                                                                              \
  /* q5 is not used , so prefetch next q5 */                                  \
  "vld1.16  {d10-d11}, [%[din_ptr0]]    \n"                                   \
  "vext.16    q14, q6, q7,  #1        \n" /* row1 input：q6，q14, q15 */    \
  "vext.16    q15, q6, q7,  #2        \n"                                     \
  "vmla.f16   q13, q6, %e[ww][0]  \n"                                         \
  "vmla.f16   q13, q14, %e[ww][1]  \n"                                        \
  "vmla.f16   q13, q15, %e[ww][2]  \n"                                        \
  "vmla.f16   q12, q6, %e[ww][3]  \n"                                         \
  "vmla.f16   q12, q14, %f[ww][0]  \n"                                        \
                                                                              \
  /* q6 is no longer uese ,so prefetch next q6 */                             \
  "vld1.16  {d12-d13}, [%[din_ptr1]]!    \n"                                  \
  "vmla.f16   q12, q15, %f[ww][1]  \n"                                        \
                                                                              \
  /* q7 is no longer uese ,so prefetch next q7 */                             \
  "vld1.16  {d14-d15}, [%[din_ptr1]]   \n"                                    \
  "vext.16    q14, q8, q9,  #1        \n" /* row2 input：q8, qq14，q15 */   \
  "vext.16    q15, q8, q9, #2        \n"                                      \
  "vmla.f16   q13, q8, %e[ww][3]  \n"                                         \
  "vmla.f16   q13, q14, %f[ww][0]  \n"                                        \
  "vmla.f16   q13, q15, %f[ww][1]  \n"                                        \
  "vmla.f16   q12, q8, %f[ww][2]  \n"                                         \
  "vmla.f16   q12, q14, %f[ww][3]  \n"                                        \
                                                                              \
  /* q8 is no longer used , so pretch next q8 */                              \
  "vld1.16  {d16-d17}, [%[din_ptr2]]!    \n"                                  \
  "vmla.f16   q12, q15, %q[ww8]  \n"                                          \
                                                                              \
  /* q9 is no longer used , so pretch next q9 */                              \
  "vld1.16  {d18-d19}, [%[din_ptr2]]    \n"                                   \
  "vext.16    q14, q10, q11, #1        \n" /* row3 input：q10，q14，q15 */ \
  "vext.16    q15, q10, q11, #2        \n"                                    \
  "vmla.f16   q13, q10, %f[ww][2]  \n"                                        \
  "vmla.f16   q13, q14, %f[ww][3]  \n"                                        \
                                                                              \
  /* q10 is no longer used , so pretch next q10 */                            \
  "vld1.16  {d20-d21}, [%[din_ptr3]]!    \n"                                  \
  "vmla.f16   q13, q15, %q[ww8]  \n"                                          \
                                                                              \
  /* q11 is no longer used , pretch next q11  */                              \
  "vld1.16  {d22-d23}, [%[din_ptr3]]    \n"                                   \
  "subs   %[cnt], %[cnt], #1           \n"

/* q12 , q13 is to hold output */
#define MID_RESULT_FP16_S1_RELU                          \
  "vmax.f16   q12,  q12, %q[vzero]              \n"      \
  "vmax.f16   q13,  q13, %q[vzero]              \n"      \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!            \n"   \
  "bne    1b                                        \n"

/* q12 , q13 is to hold output */
/* load six_ptr into q14 */
#define MID_RESULT_FP16_S1_RELU6                         \
  "vld1.16  {d28-d29}, [%[six_ptr]]    \n"               \
  "vmax.f16   q12,  q12, %q[vzero]              \n"      \
  "vmax.f16   q13,  q13, %q[vzero]              \n"      \
  "vmin.f16   q12,  q12, q14              \n"            \
  "vmin.f16   q13,  q13, q14              \n"            \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!            \n"   \
  "bne    1b                                        \n"

/* q12 , q13 is to hold output */
/* load scale_ptr into q14 first */
/* then put q15 = x / scale_ptr */
/* then put q14 = bit selector */
#define MID_RESULT_FP16_S1_LEAKY_RELU                    \
  "vld1.16  {d28-d29}, [%[scale_ptr]]    \n"             \
  "vmul.f16 q15, q12, q14 \n"                            \
  "vcge.f16 q14, q12, %q[vzero]           \n"            \
  "vbif q12, q15, q14\n"                                 \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vld1.16  {d28-d29}, [%[scale_ptr]]    \n"             \
  "vmul.f16 q15, q13, q14 \n"                            \
  "vcge.f16 q14, q13, %q[vzero]           \n"            \
  "vbif q13, q15, q14\n"                                 \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!              \n" \
  "bne    1b                                        \n"

/* q12 , q13 is to hold output */

#define MID_RESULT_FP16_S1                               \
  "vst1.32    {d24-d25}, [%[ptr_out0]]!              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]!            \n"   \
  "bne    1b                                        \n"

/* put vmask to q14 and q15，woo woo */
/* input data is in  q4-q5，q6-q7，q8-q9，q10-q11 */
/* so only q14 and q15 is avaliable for vmask */

#define RIGHT_COMPUTE_FP16_S1                                                 \
  "3:                                                     \n"                 \
  "vld1.32  {d28-d29}, [%[vmask]]! \n"                                        \
  "vld1.32  {d30-d31}, [%[vmask]] \n"                                         \
  "vld1.16    {d24-d25}, [%[bias_val]]                   \n"                  \
  "vld1.16    {d26-d27}, [%[bias_val]]                   \n"                  \
  "sub    %[din_ptr0], %[din_ptr0], %[right_pad_num]      \n"                 \
  "sub    %[din_ptr1], %[din_ptr1], %[right_pad_num]      \n"                 \
  "sub    %[din_ptr2], %[din_ptr2], %[right_pad_num]      \n"                 \
  "sub    %[din_ptr3], %[din_ptr3], %[right_pad_num]      \n"                 \
  "vld1.16  {d8-d9},   [%[din_ptr0]]!    \n"                                  \
  "vld1.16  {d12-d13}, [%[din_ptr1]]!    \n"                                  \
  "vld1.16  {d16-d17}, [%[din_ptr2]]!    \n"                                  \
  "vld1.16  {d20-d21}, [%[din_ptr3]]!    \n"                                  \
  "vld1.16  {d10-d11}, [%[din_ptr0]]    \n"                                   \
  "vld1.16  {d14-d15}, [%[din_ptr1]]   \n"                                    \
  "vld1.16  {d18-d19}, [%[din_ptr2]]    \n"                                   \
  "vld1.16  {d22-d23}, [%[din_ptr3]]    \n"                                   \
  "sub    %[ptr_out0], %[ptr_out0], %[right_st_num]       \n"                 \
  "sub    %[ptr_out1], %[ptr_out1], %[right_st_num]       \n"                 \
                                                                              \
  "vbif q4, %q[vzero], q14             \n"                                    \
  "vbif q5, %q[vzero], q15              \n"                                   \
  "vbif q6, %q[vzero], q14             \n"                                    \
  "vbif q7, %q[vzero], q15              \n"                                   \
  "vbif q8, %q[vzero], q14             \n"                                    \
  "vbif q9, %q[vzero], q15              \n"                                   \
  "vbif q10, %q[vzero], q14             \n"                                   \
  "vbif q11, %q[vzero], q15              \n"                                  \
                                                                              \
  /* the code below is same as middle */                                      \
  "vext.16    q14, q4,q5,  #1        \n" /* row0 input：q4, q14, q15 */      \
  "vext.16    q15, q4, q5,  #2        \n"                                     \
  "vmla.f16   q12, q4, %e[ww][0]  \n"                                         \
  "vmla.f16   q12, q14, %e[ww][1]  \n"                                        \
  "vmla.f16   q12, q15, %e[ww][2]  \n"                                        \
  "vext.16    q14, q6, q7,  #1        \n" /* row input：q6，q14, q15 */     \
  "vext.16    q15, q6, q7,  #2        \n"                                     \
  "vmla.f16   q13, q6, %e[ww][0]  \n"                                         \
  "vmla.f16   q13, q14, %e[ww][1]  \n"                                        \
  "vmla.f16   q13, q15, %e[ww][2]  \n"                                        \
  "vmla.f16   q12, q6, %e[ww][3]  \n"                                         \
  "vmla.f16   q12, q14, %f[ww][0]  \n"                                        \
  "vmla.f16   q12, q15, %f[ww][1]  \n"                                        \
  "vext.16    q14, q8, q9,  #1        \n" /* row2 input：q8, qq14，q15 */   \
  "vext.16    q15, q8, q9, #2        \n"                                      \
  "vmla.f16   q13, q8, %e[ww][3]  \n"                                         \
  "vmla.f16   q13, q14, %f[ww][0]  \n"                                        \
  "vmla.f16   q13, q15, %f[ww][1]  \n"                                        \
  "vmla.f16   q12, q8, %f[ww][2]  \n"                                         \
  "vmla.f16   q12, q14, %f[ww][3]  \n"                                        \
  "vmla.f16   q12, q15, %q[ww8]  \n"                                          \
  "vext.16    q14, q10, q11, #1        \n" /* row3 input：q10，q14，q15 */ \
  "vext.16    q15, q10, q11, #2        \n"                                    \
  "vmla.f16   q13, q10, %f[ww][2]  \n"                                        \
  "vmla.f16   q13, q14, %f[ww][3]  \n"                                        \
  "vmla.f16   q13, q15, %q[ww8]  \n"

/* remember only  q4, q6, q8, q10 have valid data */
/* q5, q7, q9, q11 is empty */
/*  q14 and q15 holds vmask */
#define RIGHT_COMPUTE_FP16_S1P1_SMALL                                        \
  "vld1.32  {d28-d29}, [%[vmask]]! \n"                                       \
  "vld1.16    {d24-d25}, [%[bias_val]]\n"                                    \
  "vld1.16    {d26-d27}, [%[bias_val]]\n"                                    \
                                                                             \
  /* updtae q4，6，8，10 */                                               \
  "vbif q4, %q[vzero], q14 \n"                                               \
  "vbif q6, %q[vzero], q14 \n"                                               \
  "vbif q8, %q[vzero], q14  \n"                                              \
  "vbif q10, %q[vzero], q14 \n"                                              \
                                                                             \
  /* the code below is same as left */                                       \
  "vext.16    q14, %q[vzero], q4, #7        \n" /* row0 in：q14, q4, q15 */ \
  "vext.16    q15, q4, %q[vzero],  #1        \n"                             \
  "vmla.f16   q12, q4, %e[ww][1]  \n"                                        \
  "vmla.f16   q12, q14, %e[ww][0]  \n"                                       \
  "vmla.f16   q12, q15, %e[ww][2]  \n"                                       \
  "vext.16    q14, %q[vzero], q6, #7        \n" /* row1 in q14，q6，q15 */ \
  "vext.16    q15, q6, %q[vzero],  #1        \n"                             \
  "vmla.f16   q13, q6, %e[ww][1]  \n"                                        \
  "vmla.f16   q13, q14, %e[ww][0]  \n"                                       \
  "vmla.f16   q13, q15, %e[ww][2]  \n"                                       \
  "vmla.f16   q12, q6, %f[ww][0]  \n"                                        \
  "vmla.f16   q12, q14, %e[ww][3]  \n"                                       \
  "vmla.f16   q12, q15, %f[ww][1]  \n"                                       \
  "vext.16    q14, %q[vzero], q8, #7   \n" /* row3 in: q14，q8，q15 */     \
  "vext.16    q15, q8, %q[vzero], #1    \n"                                  \
  "vmla.f16   q13, q8, %f[ww][0]  \n"                                        \
  "vmla.f16   q13, q14, %e[ww][3]  \n"                                       \
  "vmla.f16   q13, q15, %f[ww][1]  \n"                                       \
  "vmla.f16   q12, q8, %f[ww][3]  \n"                                        \
  "vmla.f16   q12, q14, %f[ww][2]  \n"                                       \
  "vmla.f16   q12, q15, %q[ww8]  \n"                                         \
  "vext.16    q14, %q[vzero], q10, #7 \n" /* row3 in: q14 10 15 */           \
  "vext.16    q15, q10, %q[vzero], #1        \n"                             \
  "vmla.f16   q13, q10, %f[ww][3]  \n"                                       \
  "vmla.f16   q13, q14, %f[ww][2]  \n"                                       \
  "vmla.f16   q13, q15, %q[ww8]  \n"

#define RIGHT_RESULT_FP16_S1_RELU                   \
  "vmax.f16   q12,  q12, %q[vzero]              \n" \
  "vmax.f16   q13,  q13, %q[vzero]              \n" \
  "vst1.32    {d24-d25}, [%[ptr_out0]]          \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]          \n"

/* q12 , q13 is to hold output */
/* load six_ptr into q14 */
#define RIGHT_RESULT_FP16_S1_RELU6                      \
  "vld1.16  {d28-d29}, [%[six_ptr]]    \n"              \
  "vmax.f16   q12,  q12, %q[vzero]              \n"     \
  "vmax.f16   q13,  q13, %q[vzero]              \n"     \
  "vmin.f16   q12,  q12, q14              \n"           \
  "vmin.f16   q13,  q13, q14              \n"           \
  "vst1.32    {d24-d25}, [%[ptr_out0]]              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]            \n"

/* q12 , q13 is to hold output */
/* load scale_ptr into q14 */
#define RIGHT_RESULT_FP16_S1_LEAKY_RELU                                     \
  "vld1.16  {d28-d29}, [%[scale_ptr]]    \n"                                \
  "vmul.f16 q15, q12, q14 \n"                 /* put q15 = x / scale_ptr */ \
  "vcge.f16 q14, q12, %q[vzero]           \n" /* put q14 = bit selector*/   \
  "vbif q12, q15, q14\n"                                                    \
  "vst1.32    {d24-d25}, [%[ptr_out0]]              \n" /* same as q12 */   \
  "vld1.16  {d28-d29}, [%[scale_ptr]]    \n"                                \
  "vmul.f16 q15, q13, q14 \n"                                               \
  "vcge.f16 q14, q13, %q[vzero]           \n"                               \
  "vbif q13, q15, q14\n"                                                    \
  "vst1.32    {d26-d27}, [%[ptr_out1]]              \n"

/* q12 , q13 is to hold output */
#define RIGHT_RESULT_FP16_S1                            \
  "vst1.32    {d24-d25}, [%[ptr_out0]]              \n" \
  "vst1.32    {d26-d27}, [%[ptr_out1]]            \n"

#endif

#define FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val) \
  float16x8_t wr = vld1q_f16(weight_ptr);            \
  float16x8_t wr8 = vdupq_n_f16(weight_ptr[8]);      \
  float16x8_t vzero = vdupq_n_f16(0.f);              \
  float16_t v_bias[8] = {bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val,                   \
                         bias_val};

#ifdef __aarch64__
#define INIT_PTR_3x3_S1_FP16(din, w_in) \
  float16_t* doutr0 = nullptr;          \
  float16_t* doutr1 = nullptr;          \
  float16_t* doutr2 = nullptr;          \
  float16_t* doutr3 = nullptr;          \
  const float16_t* dr0 = din;           \
  const float16_t* dr1 = dr0 + w_in;    \
  const float16_t* dr2 = dr1 + w_in;    \
  const float16_t* dr3 = dr2 + w_in;    \
  const float16_t* dr4 = dr3 + w_in;    \
  const float16_t* dr5 = dr4 + w_in;    \
  const float16_t* din_ptr0 = nullptr;  \
  const float16_t* din_ptr1 = nullptr;  \
  const float16_t* din_ptr2 = nullptr;  \
  const float16_t* din_ptr3 = nullptr;  \
  const float16_t* din_ptr4 = nullptr;  \
  const float16_t* din_ptr5 = nullptr;
#else

#define INIT_PTR_3x3_S1_FP16(din, w_in) \
  float16_t* doutr0 = nullptr;          \
  float16_t* doutr1 = nullptr;          \
  const float16_t* dr0 = din;           \
  const float16_t* dr1 = dr0 + w_in;    \
  const float16_t* dr2 = dr1 + w_in;    \
  const float16_t* dr3 = dr2 + w_in;    \
  const float16_t* din_ptr0 = nullptr;  \
  const float16_t* din_ptr1 = nullptr;  \
  const float16_t* din_ptr2 = nullptr;  \
  const float16_t* din_ptr3 = nullptr;

#endif

inline std::pair<uint32_t, uint32_t> right_mask_3x3s1p01_fp16(int w_in,
                                                              int w_out,
                                                              int pad,
                                                              uint16_t* vmask) {
  const uint16_t right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint32_t cnt_col = ((w_out + 7) >> 3) - 2;
  uint32_t cnt_remain = ((w_out % 8 == 0) ? 8 : w_out % 8);
  uint32_t size_right_remain =
      static_cast<uint32_t>(w_in - (w_out + 2 - pad - 10));
  uint16x8_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx));
  uint16x8_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 8));
  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);

  cnt_col = cnt_col + 1 - pad;

  return std::make_pair(cnt_col, cnt_remain);
}

inline void right_mask_3x3_s1p01_small_fp16(int w_in, uint16_t* vmask) {
  const uint16_t right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const uint16_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16_t size_right_remain = w_in;
  uint16x8_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx));
  uint16x8_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_right_remain), vld1q_u16(right_pad_idx + 8));
  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);
}

#ifdef __aarch64__

#define ASSIGN_PTR_3x3_S1_FP16(w_out) \
  din_ptr0 = dr0;                     \
  din_ptr1 = dr1;                     \
  din_ptr2 = dr2;                     \
  din_ptr3 = dr3;                     \
  din_ptr4 = dr4;                     \
  din_ptr5 = dr5;                     \
  doutr0 = dout_ptr;                  \
  doutr1 = doutr0 + w_out;            \
  doutr2 = doutr1 + w_out;            \
  doutr3 = doutr2 + w_out;

#define h_out_step 4

#else
#define ASSIGN_PTR_3x3_S1_FP16(w_out) \
  din_ptr0 = dr0;                     \
  din_ptr1 = dr1;                     \
  din_ptr2 = dr2;                     \
  din_ptr3 = dr3;                     \
  doutr0 = dout_ptr;                  \
  doutr1 = doutr0 + w_out;

#define h_out_step 2

#endif

#ifdef __aarch64__

#define TOP_BOTTOM_BORDER_3x3_S1P0_FP16(w_in, h_in, h_out) \
  dr0 = dr4;                                               \
  dr1 = dr5;                                               \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
  dr4 = dr3 + w_in;                                        \
  dr5 = dr4 + w_in;                                        \
  if (i + 5 >= h_in) {                                     \
    switch (i + 5 - h_in) {                                \
      case 4:                                              \
        din_ptr1 = zero_ptr;                               \
      case 3:                                              \
        din_ptr2 = zero_ptr;                               \
      case 2:                                              \
        din_ptr3 = zero_ptr;                               \
      case 1:                                              \
        din_ptr4 = zero_ptr;                               \
      case 0:                                              \
        din_ptr5 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 4 > h_out) {                                     \
    switch (i + 4 - h_out) {                               \
      case 3:                                              \
        doutr1 = write_ptr;                                \
      case 2:                                              \
        doutr2 = write_ptr;                                \
      case 1:                                              \
        doutr3 = write_ptr;                                \
      default:                                             \
        break;                                             \
    }                                                      \
  }

#define TOP_BOTTOM_BORDER_3x3_S1P1_FP16(w_in, h_in, h_out) \
  if (i == 0) {                                            \
    din_ptr0 = zero_ptr;                                   \
    din_ptr1 = dr0;                                        \
    din_ptr2 = dr1;                                        \
    din_ptr3 = dr2;                                        \
    din_ptr4 = dr3;                                        \
    din_ptr5 = dr4;                                        \
    dr0 = dr3;                                             \
    dr1 = dr4;                                             \
    dr2 = dr5;                                             \
    dr3 = dr2 + w_in;                                      \
    dr4 = dr3 + w_in;                                      \
    dr5 = dr4 + w_in;                                      \
  } else {                                                 \
    dr0 = dr4;                                             \
    dr1 = dr5;                                             \
    dr2 = dr1 + w_in;                                      \
    dr3 = dr2 + w_in;                                      \
    dr4 = dr3 + w_in;                                      \
    dr5 = dr4 + w_in;                                      \
  }                                                        \
  if (i + 5 > h_in) {                                      \
    switch (i + 5 - h_in) {                                \
      case 5:                                              \
        din_ptr1 = zero_ptr;                               \
      case 4:                                              \
        din_ptr2 = zero_ptr;                               \
      case 3:                                              \
        din_ptr3 = zero_ptr;                               \
      case 2:                                              \
        din_ptr4 = zero_ptr;                               \
      case 1:                                              \
        din_ptr5 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 4 > h_out) {                                     \
    switch (i + 4 - h_out) {                               \
      case 3:                                              \
        doutr1 = write_ptr;                                \
      case 2:                                              \
        doutr2 = write_ptr;                                \
      case 1:                                              \
        doutr3 = write_ptr;                                \
      default:                                             \
        break;                                             \
    }                                                      \
  }

#define SMALL_TMP_ADDR          \
  float16_t tmp_out[4][8];      \
  float16_t* tmp0 = tmp_out[0]; \
  float16_t* tmp1 = tmp_out[1]; \
  float16_t* tmp2 = tmp_out[2]; \
  float16_t* tmp3 = tmp_out[3];
#define SMALL_REAL_STORE            \
  for (int j = 0; j < w_out; j++) { \
    *(doutr0 + j) = tmp_out[0][j];  \
    *(doutr1 + j) = tmp_out[1][j];  \
    *(doutr2 + j) = tmp_out[2][j];  \
    *(doutr3 + j) = tmp_out[3][j];  \
  }

#else

#define TOP_BOTTOM_BORDER_3x3_S1P0_FP16(w_in, h_in, h_out) \
  dr0 = dr2;                                               \
  dr1 = dr3;                                               \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
                                                           \
  if (i + 3 >= h_in) {                                     \
    switch (i + 3 - h_in) {                                \
      case 2:                                              \
        din_ptr1 = zero_ptr;                               \
      case 1:                                              \
        din_ptr2 = zero_ptr;                               \
      case 0:                                              \
        din_ptr3 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 2 > h_out) {                                     \
    doutr1 = write_ptr;                                    \
  }

#define TOP_BOTTOM_BORDER_3x3_S1P1_FP16(w_in, h_in, h_out) \
  if (i == 0) {                                            \
    din_ptr0 = zero_ptr;                                   \
    din_ptr1 = dr0;                                        \
    din_ptr2 = dr1;                                        \
    din_ptr3 = dr2;                                        \
    dr0 = dr1;                                             \
    dr1 = dr2;                                             \
    dr2 = dr3;                                             \
    dr3 = dr2 + w_in;                                      \
  } else {                                                 \
    dr0 = dr2;                                             \
    dr1 = dr3;                                             \
    dr2 = dr1 + w_in;                                      \
    dr3 = dr2 + w_in;                                      \
  }                                                        \
  if (i + 3 > h_in) {                                      \
    switch (i + 3 - h_in) {                                \
      case 3:                                              \
        din_ptr1 = zero_ptr;                               \
      case 2:                                              \
        din_ptr2 = zero_ptr;                               \
      case 1:                                              \
        din_ptr3 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 2 > h_out) {                                     \
    doutr1 = write_ptr;                                    \
  }

#define SMALL_TMP_ADDR    \
  float16_t M[2][8];      \
  float16_t* tmp0 = M[0]; \
  float16_t* tmp1 = M[1];

#define SMALL_REAL_STORE            \
  for (int j = 0; j < w_out; j++) { \
    *(doutr0 + j) = tmp0[j];        \
    *(doutr1 + j) = tmp1[j];        \
  }

#endif

// w_in > 8
void conv_depthwise_3x3s1p1_bias_relu_common_fp16_fp16(float16_t* dout,
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 1, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (cnt_remain == 8) ? 16 : ((8 - cnt_remain) * 2 + 16);
  uint32_t right_st_num = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 2);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;

      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in)

          for (int i = 0; i < h_out; i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1_RELU
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww] "w"(wr), [vzero] "w"(vzero), [ww8] "w"(wr8), \
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1_RELU
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU 
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p0_bias_relu_common_fp16_fp16(float16_t* dout,
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU 
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p0_bias_relu6_common_fp16_fp16(
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU6
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU6
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [six_ptr] "r"(six), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU6 
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU6
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [six_ptr] "r"(six), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p0_bias_leaky_relu_common_fp16_fp16(
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_LEAKY_RELU
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [scale_ptr] "r"(scale), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_LEAKY_RELU 
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [scale_ptr] "r"(scale), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p0_bias_noact_common_fp16_fp16(
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else

        asm volatile(
          INIT_FP16_S1 MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1 
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p1_bias_relu6_common_fp16_fp16(
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 1, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1_RELU6
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU6
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU6
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [six_ptr] "r"(six), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1_RELU6
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_RELU6
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU6
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [six_ptr] "r"(six), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p1_bias_leaky_relu_common_fp16_fp16(
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 1, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1_LEAKY_RELU
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_LEAKY_RELU
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [scale_ptr] "r"(scale), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else

        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1_LEAKY_RELU
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1_LEAKY_RELU
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [scale_ptr] "r"(scale), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in > 8
void conv_depthwise_3x3s1p1_bias_noact_common_fp16_fp16(
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
  auto&& res = right_mask_3x3s1p01_fp16(w_in, w_out, 1, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (8 - cnt_remain) * 2 + 16;
  uint32_t right_st_num = (8 - cnt_remain) * 2;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * 9;
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [ptr_out2] "+r"(doutr2), \
              [ptr_out3] "+r"(doutr3)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num), [vmask] "r" (val_mask)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else

        asm volatile(
          INIT_FP16_S1 LEFT_COMPUTE_FP16_S1 LEFT_RESULT_FP16_S1
          MID_COMPUTE_FP16_S1 MID_RESULT_FP16_S1
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p1_bias_noact_small_fp16_fp16(float16_t* dout,
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);

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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero]"w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", \
              "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p1_bias_relu_small_fp16_fp16(float16_t* dout,
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero]"w"(vzero), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", \
              "v19", "v20", "v21", "v22"              
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );

#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p1_bias_relu6_small_fp16_fp16(float16_t* dout,
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1_RELU6
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero]"w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [six_ptr]"r"(six)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", \
              "v19", "v20", "v21", "v22"              
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1_RELU6
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [six_ptr]"r"(six)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p1_bias_leaky_relu_small_fp16_fp16(
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P1_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero]"w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [scale_ptr]"r"(scale)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", \
              "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL RIGHT_COMPUTE_FP16_S1P1_SMALL RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [scale_ptr]"r"(scale)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p0_bias_noact_small_fp16_fp16(float16_t* dout,
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
  uint32_t right_pad_num = 16;
  uint32_t right_st_num = 0;
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p0_bias_relu_small_fp16_fp16(float16_t* dout,
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
  uint32_t right_pad_num = 16;
  uint32_t right_st_num = 0;
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p0_bias_relu6_small_fp16_fp16(float16_t* dout,
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
  uint32_t right_pad_num = 16;
  uint32_t right_st_num = 0;
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in) for (int i = 0; i < h_out;
                                                  i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU6
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [six_ptr]"r"(six), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_RELU6
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [six_ptr]"r"(six), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
      }
    }
    LITE_PARALLEL_END();
  }
}

// w_in <= 8
void conv_depthwise_3x3s1p0_bias_leaky_relu_small_fp16_fp16(
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
  right_mask_3x3_s1p01_small_fp16(w_in, vmask);
  uint32_t right_pad_num = 16;
  uint32_t right_st_num = 0;
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
      FILL_WEIGHTS_BIAS_FP16(weight_ptr, bias_val)
      INIT_PTR_3x3_S1_FP16(din_ch_ptr, w_in)

          for (int i = 0; i < h_out; i += h_out_step) {
        ASSIGN_PTR_3x3_S1_FP16(w_out) TOP_BOTTOM_BORDER_3x3_S1P0_FP16(
            w_in, h_in, h_out) uint16_t* val_mask = vmask;
// clang-format off
#ifdef __aarch64__
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), [ptr_out2] "+r"(tmp2), \
              [ptr_out3] "+r"(tmp3), [vmask] "+r" (val_mask)
            : [ww]"w"(wr), [vzero] "w"(vzero), [ww8]"w"(wr8), \
              [bias_val] "r"(v_bias), [scale_ptr]"r"(scale), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", \
              "v18", "v19", "v20", "v21", "v22"
        );
#else
        asm volatile(
          INIT_FP16_S1_SMALL
          RIGHT_COMPUTE_FP16_S1 RIGHT_RESULT_FP16_S1_LEAKY_RELU
            : [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3),  \
              [ptr_out0] "+r"(tmp0), [ptr_out1] "+r"(tmp1), \
              [vmask] "+r" (val_mask)
            : [vzero] "w"(vzero),  [ww]"w"(wr), [ww8]"w"(wr8),\
              [bias_val] "r"(v_bias), [scale_ptr]"r"(scale), [right_pad_num] "r"(right_pad_num), [right_st_num] "r"(right_st_num)
            : "cc", "memory", "q4", "q5", "q6", "q7",\
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" \
        );
#endif
        // clang-format on
        SMALL_REAL_STORE
        dout_ptr += h_out_step * w_out;
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
