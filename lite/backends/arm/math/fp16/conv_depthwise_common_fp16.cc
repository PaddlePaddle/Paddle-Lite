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

#include "lite/backends/arm/math/fp16/conv_depthwise_common_fp16.h"
#include "lite/backends/arm/math/fp16/common_preprocess.h"
#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#ifdef __aarch64__
#define INIT                         \
  /*x4 = kw * dw * 8(c)*/            \
  "mul x4, %[kw], %[dilateX_step]\n" \
  "sub x5, %[dilateY_step], x4\n"    \
  "mov x7, %[weight]\n"

#define LOOP               \
  "Loop_%=:\n"             \
  "mov x9, %[pre_din]\n"   \
  "mov x10, %[pre_dout]\n" \
  "mov x6, %[ow]\n"

#define COMPUTE16                                        \
  "L16_%=:\n"                                            \
  "cmp x6, #16\n"                                        \
  "blt L8_%=\n"                                          \
  "mov x3, #16\n" /*x3 = sw * 8(c) * 16(w)*/             \
  "mul x3, %[src_w_setup], x3\n"                         \
                                                         \
  "L16Loop_%=:\n"                                        \
  "mov x8, %[pre_din]\n"                                 \
  "mov v16.16b, %[vbias].16b\n"                          \
  "mov v17.16b, %[vbias].16b\n"                          \
  "mov v18.16b, %[vbias].16b\n"                          \
  "mov v19.16b, %[vbias].16b\n"                          \
  "mov v20.16b, %[vbias].16b\n"                          \
  "mov v21.16b, %[vbias].16b\n"                          \
  "mov v22.16b, %[vbias].16b\n"                          \
  "mov v23.16b, %[vbias].16b\n"                          \
  "mov v24.16b, %[vbias].16b\n"                          \
  "mov v25.16b, %[vbias].16b\n"                          \
  "mov v26.16b, %[vbias].16b\n"                          \
  "mov v27.16b, %[vbias].16b\n"                          \
  "mov v28.16b, %[vbias].16b\n"                          \
  "mov v29.16b, %[vbias].16b\n"                          \
  "mov v30.16b, %[vbias].16b\n"                          \
  "mov v31.16b, %[vbias].16b\n"                          \
                                                         \
  /*kh*/                                                 \
  "mov x1, %[kh]\n"                                      \
  "L16LoopH_%=:\n"                                       \
  "mov x2, %[kw]\n" /*kw*/                               \
  "L16LoopW_%=:\n"                                       \
  "ld1 {v7.8h}, [%[weight]], #16\n"                      \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v16.8h, v7.8h, v0.8h\n"                          \
  "fmla v17.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v18.8h, v7.8h, v2.8h\n"                          \
  "fmla v19.8h, v7.8h, v3.8h\n"                          \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v20.8h, v7.8h, v0.8h\n"                          \
  "fmla v21.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v22.8h, v7.8h, v2.8h\n"                          \
  "fmla v23.8h, v7.8h, v3.8h\n"                          \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v24.8h, v7.8h, v0.8h\n"                          \
  "fmla v25.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v26.8h, v7.8h, v2.8h\n"                          \
  "fmla v27.8h, v7.8h, v3.8h\n"                          \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v28.8h, v7.8h, v0.8h\n"                          \
  "fmla v29.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v30.8h, v7.8h, v2.8h\n"                          \
  "fmla v31.8h, v7.8h, v3.8h\n"                          \
  "sub %[pre_din], %[pre_din], x3\n"                     \
  "add %[pre_din], %[pre_din], %[dilateX_step]\n" /*kw*/ \
  "subs x2, x2, #1\n"                                    \
  "bne L16LoopW_%=\n" /*kh*/                             \
  "add %[pre_din], %[pre_din], x5\n"                     \
  "subs x1, x1, #1\n"                                    \
  "bne L16LoopH_%=\n"                                    \
  "mov %[weight], x7\n"                                  \
  "sub x6, x6, #16\n"                                    \
  "add %[pre_din], x8, x3\n"

#define STORE16                                                \
  "st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [%[pre_dout]], #64\n" \
  "st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [%[pre_dout]], #64\n" \
  "st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [%[pre_dout]], #64\n" \
  "st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [%[pre_dout]], #64\n" \
  "cmp x6, #16\n"                                              \
  "bge L16Loop_%=\n"

#define COMPUTE8                                         \
  "L8_%=:"                                               \
  "cmp x6, #8\n"                                         \
  "blt L4_%=\n"                                          \
  "mov x3, #8\n" /*x3 = sw * 8(c) * 8(w)*/               \
  "mul x3, %[src_w_setup], x3\n"                         \
                                                         \
  "L8Loop_%=:\n"                                         \
  "mov x8, %[pre_din]\n"                                 \
  "mov v16.16b, %[vbias].16b\n"                          \
  "mov v17.16b, %[vbias].16b\n"                          \
  "mov v18.16b, %[vbias].16b\n"                          \
  "mov v19.16b, %[vbias].16b\n"                          \
  "mov v20.16b, %[vbias].16b\n"                          \
  "mov v21.16b, %[vbias].16b\n"                          \
  "mov v22.16b, %[vbias].16b\n"                          \
  "mov v23.16b, %[vbias].16b\n"                          \
                                                         \
  /*kh*/                                                 \
  "mov x1, %[kh]\n"                                      \
  "L8LoopH_%=:\n"                                        \
  "mov x2, %[kw]\n" /*kw*/                               \
  "L8LoopW_%=:\n"                                        \
  "ld1 {v7.8h}, [%[weight]], #16\n"                      \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v16.8h, v7.8h, v0.8h\n"                          \
  "fmla v17.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v18.8h, v7.8h, v2.8h\n"                          \
  "fmla v19.8h, v7.8h, v3.8h\n"                          \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v20.8h, v7.8h, v0.8h\n"                          \
  "fmla v21.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v22.8h, v7.8h, v2.8h\n"                          \
  "fmla v23.8h, v7.8h, v3.8h\n"                          \
  "sub %[pre_din], %[pre_din], x3\n"                     \
  "add %[pre_din], %[pre_din], %[dilateX_step]\n" /*kw*/ \
  "subs x2, x2, #1\n"                                    \
  "bne L8LoopW_%=\n"                                     \
  "add %[pre_din], %[pre_din], x5\n" /*kh*/              \
  "subs x1, x1, #1\n"                                    \
  "bne L8LoopH_%=\n"                                     \
  "mov %[weight], x7\n"                                  \
  "sub x6, x6, #8\n"                                     \
  "add %[pre_din], x8, x3\n"

#define STORE8                                                 \
  "st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [%[pre_dout]], #64\n" \
  "st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [%[pre_dout]], #64\n"

#define COMPUTE4                                         \
  "L4_%=:\n"                                             \
  "cmp x6, #4\n"                                         \
  "blt L1_%=\n"                                          \
  "mov x3, #4\n" /*x3 = sw * 8(c) * 4(w)*/               \
  "mul x3, %[src_w_setup], x3\n"                         \
                                                         \
  "L4Loop_%=:\n"                                         \
  "mov x8, %[pre_din]\n"                                 \
  "mov v16.16b, %[vbias].16b\n"                          \
  "mov v17.16b, %[vbias].16b\n"                          \
  "mov v18.16b, %[vbias].16b\n"                          \
  "mov v19.16b, %[vbias].16b\n" /*kh*/                   \
  "mov x1, %[kh]\n"                                      \
  "L4LoopH_%=:\n" /*kw*/                                 \
  "mov x2, %[kw]\n"                                      \
  "L4LoopW_%=:\n"                                        \
  "ld1 {v7.8h}, [%[weight]], #16\n"                      \
  "ld1 {v0.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v1.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v16.8h, v7.8h, v0.8h\n"                          \
  "fmla v17.8h, v7.8h, v1.8h\n"                          \
  "ld1 {v2.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "ld1 {v3.8h}, [%[pre_din]], %[src_w_setup]\n"          \
  "fmla v18.8h, v7.8h, v2.8h\n"                          \
  "fmla v19.8h, v7.8h, v3.8h\n"                          \
  "sub %[pre_din], %[pre_din], x3\n"                     \
  "add %[pre_din], %[pre_din], %[dilateX_step]\n" /*kw*/ \
  "subs x2, x2, #1\n"                                    \
  "bne L4LoopW_%=\n"                                     \
  "add %[pre_din], %[pre_din], x5\n" /*kh*/              \
  "subs x1, x1, #1\n"                                    \
  "bne L4LoopH_%=\n"                                     \
  "mov %[weight], x7\n"                                  \
  "sub x6, x6, #4\n"                                     \
  "add %[pre_din], x8, x3\n"

#define STORE4 "st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [%[pre_dout]], #64\n"

#define COMPUTE1                                 \
  "L1_%=:\n"                                     \
  "cmp x6, #1\n"                                 \
  "blt 0f\n"                                     \
  "L1Loop_%=:\n"                                 \
  "mov x8, %[pre_din]\n"                         \
  "mov v16.16b, %[vbias].16b\n" /*kh*/           \
  "mov x1, %[kh]\n"                              \
  "L1LoopH_%=:\n"                                \
  "mov x2, %[kw]\n" /*kw*/                       \
  "L1LoopW_%=:\n"                                \
  "ld1 {v7.8h}, [%[weight]], #16\n"              \
  "ld1 {v0.8h}, [%[pre_din]], %[dilateX_step]\n" \
  "fmla v16.8h, v7.8h, v0.8h\n" /*kw*/           \
  "subs x2, x2, #1\n"                            \
  "bne L1LoopW_%=\n" /*kh*/                      \
  "add %[pre_din], %[pre_din], x5\n"             \
  "subs x1, x1, #1\n"                            \
  "bne L1LoopH_%=\n"                             \
  "mov %[weight], x7\n"                          \
  "sub x6, x6, #1\n"                             \
  "add %[pre_din], x8, %[src_w_setup]\n"

#define STORE1                         \
  "st1 {v16.8h}, [%[pre_dout]], #16\n" \
  "cmp x6, #1\n"                       \
  "bge L1Loop_%=\n"

#define END                             \
  "0:\n"                                \
  "add %[pre_din], x9, %[srcHStep]\n"   \
  "add %[pre_dout], x10, %[dstHStep]\n" \
  "subs %[oh], %[oh], #1 \n"            \
  "bne Loop_%=\n"
#else
#define INIT                      \
  /*r10 = kw * dw * 8(c)*/        \
  "ldr r9, [%[param_ptr], #12]\n" \
  "mul r10, %[kw], r9\n"          \
  "ldr r9, [%[param_ptr], #8]\n"  \
  "sub r12, r9, r10\n"            \
  "vmov.i32 d8[0], %[weight]\n"

#define LOOP                      \
  "Loop_%=:\n"                    \
  "ldr r9, [%[param_ptr], #12]\n" \
  "push {%[oh], %[ow], %[pre_din]}\n"

#define COMPUTE8                                 \
  "L8_%=:"                                       \
  "cmp %[ow], #8\n"                              \
  "blt L4_%=\n"                                  \
  "mov %[oh], #8\n"                              \
  "mul %[oh], %[src_w_setup], %[oh]\n"           \
                                                 \
  "L8Loop_%=:\n"                                 \
  "vmov.i32 d8[1], %[pre_din]\n"                 \
  "vmov q8,  %[vbias]\n"                         \
  "vmov q9,  %[vbias]\n"                         \
  "vmov q10, %[vbias]\n"                         \
  "vmov q11, %[vbias]\n"                         \
  "vmov q12, %[vbias]\n"                         \
  "vmov q13, %[vbias]\n"                         \
  "vmov q14, %[vbias]\n"                         \
  "vmov q15, %[vbias]\n"                         \
                                                 \
  /*kh*/                                         \
  "mov r11, %[kh]\n"                             \
  "L8LoopH_%=:\n"                                \
  "mov r10, %[kw]\n" /*kw*/                      \
  "L8LoopW_%=:\n"                                \
  "vld1.16 {q3}, [%[weight]]!\n"                 \
  "vld1.16 {q0}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q8,  q0, q3\n"                       \
  "vld1.16 {q1}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q9,  q1, q3\n"                       \
  "vld1.16 {q0}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q10, q0, q3\n"                       \
  "vld1.16 {q1}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q11, q1, q3\n"                       \
  "vld1.16 {q0}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q12, q0, q3\n"                       \
  "vld1.16 {q1}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q13, q1, q3\n"                       \
  "vld1.16 {q0}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q14, q0, q3\n"                       \
  "vld1.16 {q1}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q15, q1, q3\n"                       \
  "sub %[pre_din], %[pre_din], %[oh]\n"          \
  "add %[pre_din], %[pre_din], r9\n" /*kw*/      \
  "subs r10, r10, #1\n"                          \
  "bne L8LoopW_%=\n"                             \
  "add %[pre_din], %[pre_din], r12\n" /*kh*/     \
  "subs r11, r11, #1\n"                          \
  "bne L8LoopH_%=\n"                             \
  "vmov.i32 %[weight], d8[0]\n"                  \
  "sub %[ow], %[ow], #8\n"                       \
  "vmov.i32  %[pre_din], d8[1]\n"                \
  "add %[pre_din], %[pre_din], %[oh]\n"

#define STORE8                           \
  "vst1.16  {d16-d19}, [%[pre_dout]]!\n" \
  "vst1.16  {d20-d23}, [%[pre_dout]]!\n" \
  "vst1.16  {d24-d27}, [%[pre_dout]]!\n" \
  "vst1.16  {d28-d31}, [%[pre_dout]]!\n" \
  "cmp %[ow], #8\n"                      \
  "bge L8Loop_%=\n"

#define COMPUTE4                                 \
  "L4_%=:\n"                                     \
  "cmp %[ow], #4\n"                              \
  "blt L1_%=\n"                                  \
  "mov %[oh], #4\n"                              \
  "mul %[oh], %[src_w_setup], %[oh]\n"           \
                                                 \
  "L4Loop_%=:\n"                                 \
  "vmov.i32 d8[1], %[pre_din]\n"                 \
  "vmov q8,  %[vbias]\n"                         \
  "vmov q9,  %[vbias]\n"                         \
  "vmov q10, %[vbias]\n"                         \
  "vmov q11, %[vbias]\n" /*kh*/                  \
  "mov r11, %[kh]\n"                             \
  "L4LoopH_%=:\n" /*kw*/                         \
  "mov r10, %[kw]\n"                             \
  "L4LoopW_%=:\n"                                \
  "vld1.16 {q3}, [%[weight]]!\n"                 \
  "vld1.16 {q0}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q8,  q0, q3\n"                       \
  "vld1.16 {q1}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q9,  q1, q3\n"                       \
  "vld1.16 {q0}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q10, q0, q3\n"                       \
  "vld1.16 {q1}, [%[pre_din]], %[src_w_setup]\n" \
  "vmla.f16 q11, q1, q3\n"                       \
  "sub %[pre_din], %[pre_din], %[oh]\n"          \
  "add %[pre_din], %[pre_din], r9\n" /*kw*/      \
  "subs r10, r10, #1\n"                          \
  "bne L4LoopW_%=\n"                             \
  "add %[pre_din], %[pre_din], r12\n" /*kh*/     \
  "subs r11, r11, #1\n"                          \
  "bne L4LoopH_%=\n"                             \
  "vmov.i32 %[weight], d8[0]\n"                  \
  "sub %[ow], %[ow], #4\n"                       \
  "vmov.i32 %[pre_din], d8[1]\n"                 \
  "add %[pre_din], %[pre_din], %[oh]\n"

#define STORE4                           \
  "vst1.16  {d16-d19}, [%[pre_dout]]!\n" \
  "vst1.16  {d20-d23}, [%[pre_dout]]!\n"

#define COMPUTE1                      \
  "L1_%=:\n"                          \
  "cmp %[ow], #1\n"                   \
  "blt 0f\n"                          \
  "L1Loop_%=:\n"                      \
  "vmov.i32 d8[1], %[pre_din]\n"      \
  "vmov q8,  %[vbias]\n" /*kh*/       \
  "mov r11, %[kh]\n"                  \
  "L1LoopH_%=:\n"                     \
  "mov r10, %[kw]\n" /*kw*/           \
  "L1LoopW_%=:\n"                     \
  "vld1.16 {q3}, [%[weight]]!\n"      \
  "vld1.16 {q0}, [%[pre_din]], r9\n"  \
  "vmla.f16 q8, q0, q3\n" /*kw*/      \
  "subs r10, r10, #1\n"               \
  "bne L1LoopW_%=\n" /*kh*/           \
  "add %[pre_din], %[pre_din], r12\n" \
  "subs r11, r11, #1\n"               \
  "bne L1LoopH_%=\n"                  \
  "vmov.i32 %[weight], d8[0]\n"       \
  "sub %[ow], %[ow], #1\n"            \
  "vmov.i32 %[pre_din], d8[1]\n"      \
  "add %[pre_din], %[pre_din], %[src_w_setup]\n"

#define STORE1                           \
  "vst1.16  {d16-d17}, [%[pre_dout]]!\n" \
  "cmp %[ow], #1\n"                      \
  "bge L1Loop_%=\n"

#define END                          \
  "0:\n"                             \
  "pop {%[oh], %[ow], %[pre_din]}\n" \
  "ldr r9, [%[param_ptr], #0]\n"     \
  "add %[pre_din], %[pre_din], r9\n" \
  "subs %[oh], %[oh], #1 \n"         \
  "bne Loop_%=\n"
#endif
/*
*The following function conv_depthwise_common_line is
*base on
*MNN[https://github.com/alibaba/MNN]
*
*Copyright © 2018, Alibaba Group Holding Limited
*/
void conv_depthwise_common_line(const float16_t* i_data,
                                float16_t* o_data,
                                int ic,
                                int ih,
                                int iw,
                                int bs,
                                int oc,
                                int oh,
                                int ow,
                                int kh,
                                int kw,
                                std::vector<int> strides,
                                std::vector<int> dilations,
                                std::vector<int> paddings,
                                const float16_t* weights,
                                const float16_t* bias,
                                const operators::ConvParam& param,
                                ARMContext* ctx) {
  int sh = strides[0];
  int sw = strides[1];
  int dh = dilations[0];
  int dw = dilations[1];
  int pad_top = paddings[0];
  int pad_bottom = paddings[1];
  int pad_left = paddings[2];
  int pad_right = paddings[3];
  int threads = ctx->threads();
  const int out_c_block = 8;

  const int prein_size =
      (iw + pad_left + pad_right) * (ih + pad_top + pad_bottom) * out_c_block;
  const int preout_size = ow * oh * out_c_block;
  auto workspace_size =
      threads * (prein_size + preout_size) + (iw + pad_left + pad_right);
  ctx->ExtendWorkspace(sizeof(float16_t) * workspace_size);

  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;

  /// get workspace
  auto ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, sizeof(float16_t) * (iw + pad_left + pad_right));
  float16_t* ptr_write = ptr_zero + (iw + pad_left + pad_right);
  float16_t* ptr_out =
      ptr_zero + (iw + pad_left + pad_right) + threads * prein_size;

  int size_in_channel = iw * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_left;
  int we = iw + pad_right;
  int hs = -pad_top;
  int he = ih + pad_bottom;

  uint src_w_setup = sw * out_c_block * 2;
  uint srcHStep = (iw + pad_left + pad_right) * out_c_block * sh * 2;
  uint dstHStep = ow * out_c_block * 2;
  uint dilateY_step = (iw + pad_left + pad_right) * out_c_block * dh * 2;
  uint dilateX_step = dw * out_c_block * 2;
  uint compute_param[4] = {srcHStep, dstHStep, dilateY_step, dilateX_step};
  auto param_ptr = compute_param;

  auto act_type = act_param.active_type;
  float16_t alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3 hardswish:4
  float16_t offset = 0.f;
  float16_t threshold = 6.f;

  if (act_param.has_active) {
    act_acquire(act_type, flag_act, alpha, offset, threshold, act_param);
  }

  for (int n = 0; n < bs; ++n) {
    const float16_t* din_batch = i_data + n * ic * size_in_channel;
    float16_t* dout_batch = o_data + n * oc * size_out_channel;
    LITE_PARALLEL_COMMON_BEGIN(c, tid, oc, 0, out_c_block) {
      auto oh_bak = oh;
#ifdef LITE_USE_THREAD_POOL
      float16_t* pre_din = ptr_write + tid * prein_size;
      float16_t* pre_dout = ptr_out + tid * preout_size;
#elif ARM_WITH_OMP
      float16_t* pre_din = ptr_write + omp_get_thread_num() * prein_size;
      float16_t* pre_dout = ptr_out + omp_get_thread_num() * preout_size;
#else
      float16_t* pre_din = ptr_write;
      float16_t* pre_dout = ptr_out;
#endif
      float16_t* out_back = pre_dout;
      prepack_input_nxwc8_fp16_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, iw, ih, ptr_zero);
      const float16_t* weight_c = weights + c * kw * kh;  // kernel_w * kernel_h
      float16_t bias_local[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      float16x8_t vbias = vdupq_n_f16(0.f);
      if (flag_bias) {
        if (c + out_c_block < oc) {
          vbias = vld1q_f16(&bias[c]);
        } else {
          for (int k = 0; k < 8 && c + k < oc; k++) {
            bias_local[k] = bias[c + k];
          }
          vbias = vld1q_f16(bias_local);
        }
      }
#ifdef __aarch64__
      asm volatile(INIT LOOP COMPUTE16 STORE16 COMPUTE8 STORE8 COMPUTE4 STORE4
                       COMPUTE1 STORE1 END
                   : [pre_dout] "+r"(pre_dout),
                     [pre_din] "+r"(pre_din),
                     [weight] "+r"(weight_c),
                     [oh] "+r"(oh_bak)
                   : [ow] "r"(ow),
                     [src_w_setup] "r"(src_w_setup),
                     [kh] "r"(kh),
                     [kw] "r"(kw),
                     [dilateY_step] "r"(dilateY_step),
                     [dilateX_step] "r"(dilateX_step),
                     [srcHStep] "r"(srcHStep),
                     [dstHStep] "r"(dstHStep),
                     [vbias] "w"(vbias)
                   : "cc",
                     "memory",
                     "x1",
                     "x2",
                     "x3",
                     "x4",
                     "x5",
                     "x6",
                     "x7",
                     "x8",
                     "x9",
                     "x10",
                     "v0",
                     "v1",
                     "v2",
                     "v3",
                     "v4",
                     "v5",
                     "v6",
                     "v7",
                     "v8",
                     "v16",
                     "v17",
                     "v18",
                     "v19",
                     "v20",
                     "v21",
                     "v22",
                     "v23",
                     "v24",
                     "v25",
                     "v26",
                     "v27",
                     "v28",
                     "v29",
                     "v30",
                     "v31");
#else
      asm volatile(INIT LOOP COMPUTE8 STORE8 COMPUTE4 STORE4 COMPUTE1 STORE1 END
                   : [pre_dout] "+r"(pre_dout),
                     [pre_din] "+r"(pre_din),
                     [weight] "+r"(weight_c),
                     [oh] "+r"(oh_bak)
                   : [ow] "r"(ow),
                     [src_w_setup] "r"(src_w_setup),
                     [kh] "r"(kh),
                     [kw] "r"(kw),
                     [param_ptr] "r"(param_ptr),
                     [vbias] "w"(vbias)
                   : "cc",
                     "memory",
                     "r9",
                     "r10",
                     "r11",
                     "r12",
                     "q0",
                     "q1",
                     "q3",
                     "q4",
                     "q8",
                     "q9",
                     "q10",
                     "q11",
                     "q12",
                     "q13",
                     "q14",
                     "q15");
#endif
      write_to_oc8_fp16(out_back,
                        dout_batch,
                        c,
                        c + out_c_block,
                        0,
                        oh,
                        0,
                        ow,
                        oc,
                        oh,
                        ow,
                        flag_act,
                        alpha,
                        nullptr,
                        false,
                        offset,
                        threshold);
    }
    LITE_PARALLEL_COMMON_END()
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
