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

#else  // armv7

#define SPARSE_F32_F32_W48_v7_KERNEL \
  "vdup.32    q4,    %[vbias]\n"     \
  "vdup.32    q5,    d8[0]\n"        \
  "vdup.32    q6,    d8[0]\n"        \
  "pld  [%[a_ptr], #128]    \n"      \
  "vdup.32    q7,    d8[0]\n"        \
  "vdup.32    q8,    d8[0]\n"        \
  "vdup.32    q9,    d8[0]\n"        \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vdup.32    q10,   d8[0]\n"        \
  "vdup.32    q11,   d8[0]\n"        \
  "vdup.32    q12,   d8[0]\n"        \
  "pld  [%[b_ptr], #192]    \n"      \
  "vdup.32    q13,   d8[0]\n"        \
  "vdup.32    q14,   d8[0]\n"        \
  "vdup.32    q15,   d8[0]\n"        \
  "cmp    %[k],    #0\n"             \
  "beq    1f\n" /* main loop*/       \
  "0:\n"                             \
  "ldr   r0, [%[a_ptr]], #4\n"       \
  "mov   r2,   %[b_ptr]\n"           \
  "vdup.32    q0,   r0\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "vmla.f32    q4,   q1,  q0\n"      \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vmla.f32    q5,   q2,  q0\n"      \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "vmla.f32    q6,  q1,  q0\n"       \
  "vmla.f32    q7,  q2,  q0\n"       \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "vmla.f32    q8,  q1,  q0\n"       \
  "vmla.f32    q9,  q2,  q0\n"       \
  "ldr   r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"  \
  "subs    %[k],   %[k],   #1\n"     \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "vmla.f32    q10,  q1,  q0\n"      \
  "vmla.f32    q11,  q2,  q0\n"      \
  "pld  [%[b_ptr], #192]    \n"      \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "vmla.f32    q12,  q1,  q0\n"      \
  "vmla.f32    q13,  q2,  q0\n"      \
  "pld  [%[a_ptr], #128]    \n"      \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "vmla.f32    q14,  q1,  q0\n"      \
  "vmla.f32    q15,  q2,  q0\n"      \
  "bne     0b\n"                     \
  "1:\n"

#define SPARSE_F32_F32_W48_v7_RELU               \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i32   q0, #0\n"         /* for relu */   \
  "vmax.f32   q4,   q4,   q0\n" /* relu */       \
  "vmax.f32   q5,   q5,   q0\n" /* relu */       \
  "vmax.f32   q6,   q6,   q0\n" /* relu */       \
  "vmax.f32   q7,   q7,   q0\n" /* relu */       \
  "vmax.f32   q8,   q8,   q0\n" /* relu */       \
  "vmax.f32   q9,   q9,   q0\n" /* relu */       \
  "vmax.f32   q10,  q10,  q0\n" /* relu */       \
  "vmax.f32   q11,  q11,  q0\n" /* relu */       \
  "vmax.f32   q12,  q12,  q0\n" /* relu */       \
  "vmax.f32   q13,  q13,  q0\n" /* relu */       \
  "vmax.f32   q14,  q14,  q0\n" /* relu */       \
  "vmax.f32   q15,  q15,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F32_F32_W48_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q4,   q4,   q0\n"      /* relu6 */       \
  "vmax.f32   q5,   q5,   q0\n"      /* relu6 */       \
  "vmax.f32   q6,   q6,   q0\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu6 */       \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f32   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f32   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f32   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f32   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q1\n"      /* relu6 */       \
  "vmin.f32   q5,   q5,   q1\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q1\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q1\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f32   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f32   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f32   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f32   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W48_v7_LEAKY_RELU                       \
  /* do relu */                                                \
  "11: \n"                                                     \
  "vmov.i32   q0, #0\n"                /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f32   q2,    q4,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q4,    q1     \n" /* vmulq_f32 */         \
  "vbif       q4,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q5,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q5,    q1     \n" /* vmulq_f32 */         \
  "vbif       q5,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q6,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q6,    q1     \n" /* vmulq_f32 */         \
  "vbif       q6,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q7,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q7,    q1     \n" /* vmulq_f32 */         \
  "vbif       q7,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q8,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1     \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q9,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q9,    q1     \n" /* vmulq_f32 */         \
  "vbif       q9,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q11,    q1    \n" /* vmulq_f32 */         \
  "vbif       q11,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q12,    q1    \n" /* vmulq_f32 */         \
  "vbif       q12,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q13,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q14,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q14,    q1    \n" /* vmulq_f32 */         \
  "vbif       q14,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q15,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q15,    q1    \n" /* vmulq_f32 */         \
  "vbif       q15,    q3,    q2    \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_F32_F32_W48_v7_OUT                                  \
  SPARSE_F32_F32_W48_v7_KERNEL SPARSE_F32_F32_W48_v7_RELU          \
      SPARSE_F32_F32_W48_v7_RELU6 SPARSE_F32_F32_W48_v7_LEAKY_RELU \
      "mov   r2,   %[c_ptr]\n" /* store result */                  \
      "vst1.32   {d8-d11},  [r2]\n"                                \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d12-d15},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d16-d19},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d20-d23},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d24-d27},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d28-d31},  [r2]\n"

#define SPARSE_F32_F32_W32_v7_KERNEL \
  "vdup.32    q8,   %[vbias]\n"      \
  "vdup.32    q9,   d16[0]\n"        \
  "pld  [%[a_ptr], #128]    \n"      \
  "vdup.32    q10,  d16[0]\n"        \
  "vdup.32    q11,  d16[0]\n"        \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vdup.32    q12,  d16[0]\n"        \
  "vdup.32    q13,  d16[0]\n"        \
  "pld  [%[b_ptr], #128]    \n"      \
  "vdup.32    q14,  d16[0]\n"        \
  "vdup.32    q15,  d16[0]\n"        \
  "cmp    %[k],    #0\n"             \
  "beq    1f\n" /* main loop*/       \
  "0:\n"                             \
  "ldr   r0, [%[a_ptr]], #4\n"       \
  "mov   r2,   %[b_ptr]\n"           \
  "vdup.32    q0,   r0\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d6-d9}, [r2]\n"         \
  "vmla.f32    q8,   q1,  q0\n"      \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vmla.f32    q9,   q2,  q0\n"      \
  "vmla.f32    q10,  q3,  q0\n"      \
  "vmla.f32    q11,  q4,  q0\n"      \
  "ldr   r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"  \
  "subs    %[k],   %[k],   #1\n"     \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "pld  [%[b_ptr], #128]    \n"      \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d6-d9}, [r2]\n"         \
  "vmla.f32    q12,  q1,  q0\n"      \
  "pld  [%[a_ptr], #128]    \n"      \
  "vmla.f32    q13,  q2,  q0\n"      \
  "vmla.f32    q14,  q3,  q0\n"      \
  "vmla.f32    q15,  q4,  q0\n"      \
  "bne     0b\n"                     \
  "1:\n"

#define SPARSE_F32_F32_W32_v7_RELU               \
  /* do relu */                                  \
  "cmp    %[vflag_act],   #0\n" /* skip relu */  \
  "beq   9f                 \n" /* no act end */ \
  "cmp    %[vflag_act],   #1\n" /* skip relu */  \
  "bne   10f                \n" /* other act */  \
  "vmov.i32   q0, #0\n"         /* for relu */   \
  "vmax.f32   q8,   q8,   q0\n" /* relu */       \
  "vmax.f32   q9,   q9,   q0\n" /* relu */       \
  "vmax.f32   q10,  q10,  q0\n" /* relu */       \
  "vmax.f32   q11,  q11,  q0\n" /* relu */       \
  "vmax.f32   q12,  q12,  q0\n" /* relu */       \
  "vmax.f32   q13,  q13,  q0\n" /* relu */       \
  "vmax.f32   q14,  q14,  q0\n" /* relu */       \
  "vmax.f32   q15,  q15,  q0\n" /* relu */       \
  "b      9f                \n" /* relu end */

#define SPARSE_F32_F32_W32_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f32   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f32   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f32   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f32   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f32   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f32   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f32   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f32   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W32_v7_LEAKY_RELU                      \
  /* do relu */                                               \
  "11: \n"                                                    \
  "vmov.i32   q0, #0\n"               /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"       /* leakey relu alpha */ \
  "vcge.f32   q2,    q8,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1    \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q9,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q9,    q1    \n" /* vmulq_f32 */         \
  "vcge.f32   q6,    q10,   q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q7,    q10,   q1    \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2    \n"                         \
  "vbif       q9,    q5,    q4    \n"                         \
  "vbif       q10,   q7,    q6    \n"                         \
  "vcge.f32   q2,    q11,   q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q11,   q1    \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q12,   q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q12,   q1    \n" /* vmulq_f32 */         \
  "vcge.f32   q6,    q13,   q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q7,    q13,   q1    \n" /* vmulq_f32 */         \
  "vbif       q11,   q3,    q2    \n"                         \
  "vbif       q12,   q5,    q4    \n"                         \
  "vbif       q13,   q7,    q6    \n"                         \
  "vcge.f32   q2,    q14,   q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q14,   q1    \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q15,   q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q15,   q1    \n" /* vmulq_f32 */         \
  "vbif       q14,   q3,    q2    \n"                         \
  "vbif       q15,   q5,    q4    \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_F32_F32_W32_v7_OUT                                  \
  SPARSE_F32_F32_W32_v7_KERNEL SPARSE_F32_F32_W32_v7_RELU          \
      SPARSE_F32_F32_W32_v7_RELU6 SPARSE_F32_F32_W32_v7_LEAKY_RELU \
      "mov   r2,   %[c_ptr]\n" /* store result */                  \
      "vst1.32   {d16-d19},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d20-d23},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d24-d27},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d28-d31},  [r2]\n"

#define SPARSE_F32_F32_W16_v7_KERNEL \
  "vdup.32    q8,   %[vbias]\n"      \
  "pld  [%[a_ptr], #128]    \n"      \
  "vdup.32    q9,   d16[0]\n"        \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vdup.32    q10,  d16[0]\n"        \
  "pld  [%[b_ptr], #128]    \n"      \
  "vdup.32    q11,  d16[0]\n"        \
  "cmp    %[k],    #0\n"             \
  "beq    1f\n" /* main loop*/       \
  "0:\n"                             \
  "ldr   r0, [%[a_ptr]], #4\n"       \
  "subs    %[k],   %[k],   #1\n"     \
  "mov   r2,   %[b_ptr]\n"           \
  "pld  [%[widx_dmap], #128]    \n"  \
  "vdup.32    q0,   r0\n"            \
  "vld1.32  {d2-d5}, [r2]\n"         \
  "add  r2,  r2,   #32\n"            \
  "vld1.32  {d6-d9}, [r2]\n"         \
  "ldr   r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"  \
  "vmla.f32    q8,   q1,  q0\n"      \
  "vmla.f32    q9,   q2,  q0\n"      \
  "pld  [%[b_ptr], #128]    \n"      \
  "vmla.f32    q10,  q3,  q0\n"      \
  "vmla.f32    q11,  q4,  q0\n"      \
  "bne     0b\n"                     \
  "1:\n"

#define SPARSE_F32_F32_W16_v7_RELU                    \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q8,   q8,   q0\n"      /* relu */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W16_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W16_v7_LEAKY_RELU                      \
  /* do relu */                                               \
  "11: \n"                                                    \
  "vmov.i32   q0, #0\n"               /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"       /* leakey relu alpha */ \
  "vcge.f32   q2,    q8,    q0   \n"  /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1   \n"  /* vmulq_f32 */         \
  "vcge.f32   q4,    q9,    q0   \n"  /* vcgeq_f32 */         \
  "vmul.f32   q5,    q9,    q1   \n"  /* vmulq_f32 */         \
  "vcge.f32   q6,    q10,   q0   \n"  /* vcgeq_f32 */         \
  "vmul.f32   q7,    q10,   q1   \n"  /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2   \n"  /* vmulq_f32 */         \
  "vbif       q9,    q5,    q4   \n"  /* vmulq_f32 */         \
  "vbif       q10,   q7,    q6   \n"  /* vmulq_f32 */         \
  "vcge.f32   q2,    q11,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q11,    q1   \n" /* vmulq_f32 */         \
  "vbif       q11,   q3,    q2   \n"  /* vmulq_f32 */         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_F32_F32_W16_v7_OUT                                  \
  SPARSE_F32_F32_W16_v7_KERNEL SPARSE_F32_F32_W16_v7_RELU          \
      SPARSE_F32_F32_W16_v7_RELU6 SPARSE_F32_F32_W16_v7_LEAKY_RELU \
      "mov   r2,   %[c_ptr]\n" /* store result */                  \
      "vst1.32   {d16-d19},  [r2]\n"                               \
      "add  r2,  r2,   #32\n"                                      \
      "vst1.32   {d20-d23},  [r2]\n"

#define SPARSE_F32_F32_W8_v7_KERNEL \
  "vdup.32    q3,   %[vbias]\n"     \
  "vdup.32    q4,   d6[0]\n"        \
  "cmp    %[k],    #0\n"            \
  "beq    1f\n" /* main loop*/      \
  "0:\n"                            \
  "ldr   r0, [%[a_ptr]], #4\n"      \
  "vdup.32    q0,   r0\n"           \
  "vld1.32  {d2-d5}, [%[b_ptr]]\n"  \
  "vmla.f32    q3,   q1,  q0\n"     \
  "vmla.f32    q4,   q2,  q0\n"     \
  "ldr   r1, [%[widx_dmap]], #4\n"  \
  "add   %[b_ptr],  %[b_ptr], r1\n" \
  "subs    %[k],   %[k],   #1\n"    \
  "bne     0b\n"                    \
  "1:\n"

#define SPARSE_F32_F32_W8_v7_RELU                     \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q3,   q3,   q0\n"      /* relu */       \
  "vmax.f32   q4,   q4,   q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W8_v7_RELU6                     \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q3,   q3,   q0\n"      /* relu6 */       \
  "vmax.f32   q4,   q4,   q0\n"      /* relu6 */       \
  "vmin.f32   q3,   q3,   q1\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W8_v7_LEAKY_RELU                      \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q5,    q3,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q6,    q3,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q7,    q4,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q8,    q4,    q1   \n" /* vmulq_f32 */         \
  "vbif       q3,    q6,    q5   \n" /* vmulq_f32 */         \
  "vbif       q4,    q8,    q7   \n" /* vmulq_f32 */         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_F32_F32_W8_v7_OUT                             \
  SPARSE_F32_F32_W8_v7_KERNEL SPARSE_F32_F32_W8_v7_RELU      \
      SPARSE_F32_F32_W8_v7_RELU6                             \
          SPARSE_F32_F32_W8_v7_LEAKY_RELU /* store result */ \
      "vst1.32   {d6-d9},  [%[c_ptr]]\n"

#define SPARSE_F32_F32_W4_v7_KERNEL \
  "vdup.32    q3,   %[vbias]\n"     \
  "cmp    %[k],    #0\n"            \
  "beq    1f\n" /* main loop*/      \
  "0:\n"                            \
  "ldr   r0, [%[a_ptr]], #4\n"      \
  "subs    %[k],   %[k],   #1\n"    \
  "vdup.32    q0,   r0\n"           \
  "vld1.32  {d2-d3}, [%[b_ptr]]\n"  \
  "vmla.f32    q3,   q1,  q0\n"     \
  "ldr   r1, [%[widx_dmap]], #4\n"  \
  "add   %[b_ptr],  %[b_ptr], r1\n" \
  "bne     0b\n"                    \
  "1:\n"

#define SPARSE_F32_F32_W4_v7_RELU                     \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q3,   q3,   q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W4_v7_RELU6                     \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q3,   q3,   q0\n"      /* relu6 */       \
  "vmin.f32   q3,   q3,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_F32_F32_W4_v7_LEAKY_RELU                      \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q4,    q3,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q3,    q1   \n" /* vmulq_f32 */         \
  "vbif       q3,    q5,    q4   \n" /* vmulq_f32 */         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_F32_F32_W4_v7_OUT                             \
  SPARSE_F32_F32_W4_v7_KERNEL SPARSE_F32_F32_W4_v7_RELU      \
      SPARSE_F32_F32_W4_v7_RELU6                             \
          SPARSE_F32_F32_W4_v7_LEAKY_RELU /* store result */ \
      "vst1.32   {d6-d7},  [%[c_ptr]]\n"

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
        uint32_t pair_num = nnz % 4;
        uint32_t lave_num = (pair_num == 0) ? 0 : (4 - pair_num);
        float vbias = (bias != nullptr) ? bias[i] : 0.0;
        // clang-format off
            asm volatile(SPARSE_F32_F32_W48_v7_OUT  
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
                "r2",
                "cc",
                "memory");
        // clang-format on
        output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        w = w + lave_num;
        dmap = dmap + lave_num;
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
          uint32_t pair_num = nnz % 4;
          uint32_t lave_num = (pair_num == 0) ? 0 : (4 - pair_num);
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W32_v7_OUT  
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
                "r0",
                "r1",
                "r2",
                "cc",
                "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
          w = w + lave_num;
          dmap = dmap + lave_num;
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
          uint32_t pair_num = nnz % 4;
          uint32_t lave_num = (pair_num == 0) ? 0 : (4 - pair_num);
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W16_v7_OUT  
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
                "r0",
                "r1",
                "r2",
                "cc",
                "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
          w = w + lave_num;
          dmap = dmap + lave_num;
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
          uint32_t pair_num = nnz % 4;
          uint32_t lave_num = (pair_num == 0) ? 0 : (4 - pair_num);
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W8_v7_OUT  
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
              : "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "r0",
                "r1",
                "cc",
                "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
          w = w + lave_num;
          dmap = dmap + lave_num;
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
          uint32_t pair_num = nnz % 4;
          uint32_t lave_num = (pair_num == 0) ? 0 : (4 - pair_num);
          float vbias = (bias != nullptr) ? bias[i] : 0.0;
          // clang-format off
            asm volatile(SPARSE_F32_F32_W4_v7_OUT  
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
              : "q0",
                "q1",
                "q3",
                "q4",
                "q5",
                "r0",
                "r1",
                "cc",
                "memory");
          // clang-format on
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
          w = w + lave_num;
          dmap = dmap + lave_num;
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

#define SPARSE_INT8_F32_W48_v7_KERNEL  \
  "veor   q4,   q0,  q0\n"             \
  "veor   q5,   q1,  q1\n"             \
  "veor   q6,   q2,  q2\n"             \
  "veor   q7,   q3,  q3\n"             \
  "pld  [%[a_ptr], #32]    \n"         \
  "veor   q8,   q0,  q0\n"             \
  "veor   q9,   q1,  q1\n"             \
  "pld  [%[widx_dmap], #32]    \n"     \
  "veor   q10,  q2,  q2\n"             \
  "veor   q11,  q3,  q3\n"             \
  "veor   q12,  q0,  q0\n"             \
  "pld  [%[b_ptr], #64]    \n"         \
  "veor   q13,  q1,  q1\n"             \
  "veor   q14,  q2,  q2\n"             \
  "veor   q15,  q3,  q3\n"             \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "mov   r2,   %[b_ptr]\n"             \
  "vld1.8  {d2-d5}, [r2]\n"            \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q3,    d2,  d0\n"       \
  "vaddw.s16   q4,    q4,  d6\n"       \
  "vaddw.s16   q5,    q5,  d7\n"       \
  "vmull.s8    q3,    d3,  d0\n"       \
  "pld  [%[widx_dmap], #32]    \n"     \
  "vaddw.s16   q6,    q6,  d6\n"       \
  "vaddw.s16   q7,    q7,  d7\n"       \
  "vmull.s8    q3,    d4,  d0\n"       \
  "vaddw.s16   q8,    q8, d6\n"        \
  "vaddw.s16   q9,    q9, d7\n"        \
  "vmull.s8    q3,    d5,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q10,   q10, d6\n"       \
  "vaddw.s16   q11,   q11, d7\n"       \
  "pld  [%[b_ptr], #64]    \n"         \
  "add  r2,  r2,   #32\n"              \
  "vld1.8  {d2-d3}, [r2]\n"            \
  "vmull.s8    q3,    d2,  d0\n"       \
  "vaddw.s16   q12,   q12,  d6\n"      \
  "pld  [%[a_ptr], #32]    \n"         \
  "vaddw.s16   q13,   q13,  d7\n"      \
  "vmull.s8    q3,    d3,  d0\n"       \
  "vaddw.s16   q14,   q14,  d6\n"      \
  "vaddw.s16   q15,   q15,  d7\n"      \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_FP32_W48_v7      \
  /* write output */                                \
  "vdup.32    q0,   %[vscale]\n"                    \
  "vdup.32    q1,   %[vbias]\n"                     \
  "vcvt.f32.s32   q2, q4\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q5\n" /* cvt int32 to fp32*/  \
  "vdup.32    q4,   d2[0]\n"                        \
  "vdup.32    q5,   d2[0]\n"                        \
  "vmla.f32   q4,  q2,  q0\n"                       \
  "vmla.f32   q5,  q3,  q0\n"                       \
  "vcvt.f32.s32   q2, q6\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q7\n" /* cvt int32 to fp32*/  \
  "vdup.32    q6,   d2[0]\n"                        \
  "vdup.32    q7,   d2[0]\n"                        \
  "vmla.f32   q6,  q2,  q0\n"                       \
  "vmla.f32   q7,  q3,  q0\n"                       \
  "vcvt.f32.s32   q2, q8\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q9\n" /* cvt int32 to fp32*/  \
  "vdup.32    q8,   d2[0]\n"                        \
  "vdup.32    q9,   d2[0]\n"                        \
  "vmla.f32   q8,  q2,  q0\n"                       \
  "vmla.f32   q9,  q3,  q0\n"                       \
  "vcvt.f32.s32   q2, q10\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q11\n" /* cvt int32 to fp32*/ \
  "vdup.32    q10,   d2[0]\n"                       \
  "vdup.32    q11,   d2[0]\n"                       \
  "vmla.f32   q10,  q2,  q0\n"                      \
  "vmla.f32   q11,  q3,  q0\n"                      \
  "vcvt.f32.s32   q2, q12\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q13\n" /* cvt int32 to fp32*/ \
  "vdup.32    q12,   d2[0]\n"                       \
  "vdup.32    q13,   d2[0]\n"                       \
  "vmla.f32   q12,  q2,  q0\n"                      \
  "vmla.f32   q13,  q3,  q0\n"                      \
  "vcvt.f32.s32   q2, q14\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q15\n" /* cvt int32 to fp32*/ \
  "vdup.32    q14,   d2[0]\n"                       \
  "vdup.32    q15,   d2[0]\n"                       \
  "vmla.f32   q14,  q2,  q0\n"                      \
  "vmla.f32   q15,  q3,  q0\n"

#define SPARSE_INT8_F32_W48_v7_RELU                   \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q4,   q4,   q0\n"      /* relu */       \
  "vmax.f32   q5,   q5,   q0\n"      /* relu */       \
  "vmax.f32   q6,   q6,   q0\n"      /* relu */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu */       \
  "vmax.f32   q8,   q8,   q0\n"      /* relu */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu */       \
  "vmax.f32   q12,  q12,  q0\n"      /* relu */       \
  "vmax.f32   q13,  q13,  q0\n"      /* relu */       \
  "vmax.f32   q14,  q14,  q0\n"      /* relu */       \
  "vmax.f32   q15,  q15,  q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W48_v7_RELU6                   \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q4,   q4,   q0\n"      /* relu6 */       \
  "vmax.f32   q5,   q5,   q0\n"      /* relu6 */       \
  "vmax.f32   q6,   q6,   q0\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu6 */       \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f32   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f32   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f32   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f32   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q1\n"      /* relu6 */       \
  "vmin.f32   q5,   q5,   q1\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q1\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q1\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f32   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f32   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f32   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f32   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W48_v7_LEAKY_RELU                      \
  /* do relu */                                                \
  "11: \n"                                                     \
  "vmov.i32   q0, #0\n"                /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f32   q2,    q4,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q4,    q1     \n" /* vmulq_f32 */         \
  "vbif       q4,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q5,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q5,    q1     \n" /* vmulq_f32 */         \
  "vbif       q5,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q6,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q6,    q1     \n" /* vmulq_f32 */         \
  "vbif       q6,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q7,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q7,    q1     \n" /* vmulq_f32 */         \
  "vbif       q7,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q8,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1     \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q9,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q9,    q1     \n" /* vmulq_f32 */         \
  "vbif       q9,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q11,    q1    \n" /* vmulq_f32 */         \
  "vbif       q11,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q12,    q1    \n" /* vmulq_f32 */         \
  "vbif       q12,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q13,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q14,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q14,    q1    \n" /* vmulq_f32 */         \
  "vbif       q14,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q15,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q15,    q1    \n" /* vmulq_f32 */         \
  "vbif       q15,    q3,    q2    \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_INT8_F32_W48_v7_OUT                                     \
  SPARSE_INT8_F32_W48_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_FP32_W48_v7 \
      SPARSE_INT8_F32_W48_v7_RELU SPARSE_INT8_F32_W48_v7_RELU6         \
          SPARSE_INT8_F32_W48_v7_LEAKY_RELU                            \
      "mov   r0,   %[c_ptr]\n" /* store result */                      \
      "vst1.32   {d8-d11},  [r0]\n"                                    \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d12-d15},  [r0]\n"                                   \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d16-d19},  [r0]\n"                                   \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d20-d23},  [r0]\n"                                   \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d24-d27},  [r0]\n"                                   \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d28-d31},  [r0]\n"

#define SPARSE_INT8_F32_W32_v7_KERNEL  \
  "veor   q8,   q0,  q0\n"             \
  "veor   q9,   q1,  q1\n"             \
  "pld  [%[a_ptr], #32]    \n"         \
  "veor   q10,  q2,  q2\n"             \
  "veor   q11,  q3,  q3\n"             \
  "pld  [%[widx_dmap], #32]    \n"     \
  "veor   q12,  q4,  q4\n"             \
  "veor   q13,  q5,  q5\n"             \
  "pld  [%[b_ptr], #64]    \n"         \
  "veor   q14,  q6,  q6\n"             \
  "veor   q15,  q7,  q7\n"             \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "vld1.8  {d2-d5}, [%[b_ptr]]\n"      \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q3,    d2,  d0\n"       \
  "vmull.s8    q4,    d3,  d0\n"       \
  "pld  [%[widx_dmap], #32]    \n"     \
  "vmull.s8    q5,    d4,  d0\n"       \
  "vmull.s8    q6,    d5,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q8,     q8,  d6\n"      \
  "vaddw.s16   q9,     q9,  d7\n"      \
  "pld  [%[a_ptr], #32]    \n"         \
  "vaddw.s16   q10,    q10,  d8\n"     \
  "vaddw.s16   q11,    q11,  d9\n"     \
  "vaddw.s16   q12,    q12,  d10\n"    \
  "pld  [%[b_ptr], #64]    \n"         \
  "vaddw.s16   q13,    q13,  d11\n"    \
  "vaddw.s16   q14,    q14,  d12\n"    \
  "vaddw.s16   q15,    q15,  d13\n"    \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_FP32_W32_v7       \
  /* write output */                                 \
  "vcvt.f32.s32   q0, q8\n"  /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q1, q9\n"  /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q2, q10\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q11\n" /* cvt int32 to fp32*/  \
  "vdup.32    q8,   %[vscale]\n"                     \
  "vdup.32    q4,   %[vbias]\n"                      \
  "vdup.32    q5,   d8[0]\n"                         \
  "vdup.32    q6,   d8[0]\n"                         \
  "vdup.32    q7,   d8[0]\n"                         \
  "3:\n"                                             \
  "vmla.f32  q4,  q0,  q8\n"                         \
  "vmla.f32  q5,  q1,  q8\n"                         \
  "vmla.f32  q6,  q2,  q8\n"                         \
  "vmla.f32  q7,  q3,  q8\n"                         \
  "4:\n"                                             \
  "vcvt.f32.s32   q8, q12\n"  /* int32 to fp32*/     \
  "vcvt.f32.s32   q9, q13\n"  /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q10, q14\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q11, q15\n" /* cvt int32 to fp32*/ \
  "vdup.32    q12,   %[vscale]\n"                    \
  "vdup.32    q0,   %[vbias]\n"                      \
  "vdup.32    q1,   d0[0]\n"                         \
  "vdup.32    q2,   d0[0]\n"                         \
  "vdup.32    q3,   d0[0]\n"                         \
  "6:\n"                                             \
  "vmla.f32  q0,  q8,  q12\n"                        \
  "vmla.f32  q1,  q9,  q12\n"                        \
  "vmla.f32  q2,  q10,  q12\n"                       \
  "vmla.f32  q3,  q11,  q12\n"

#define SPARSE_INT8_F32_W32_v7_RELU                   \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q8, #0\n"              /* for relu */   \
  "vmax.f32   q0,   q0,   q8\n"      /* relu */       \
  "vmax.f32   q1,   q1,   q8\n"      /* relu */       \
  "vmax.f32   q2,   q2,   q8\n"      /* relu */       \
  "vmax.f32   q3,   q3,   q8\n"      /* relu */       \
  "vmax.f32   q4,   q4,   q8\n"      /* relu */       \
  "vmax.f32   q5,   q5,   q8\n"      /* relu */       \
  "vmax.f32   q6,   q6,   q8\n"      /* relu */       \
  "vmax.f32   q7,   q7,   q8\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W32_v7_RELU6                   \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q8,   #0\n"            /* for relu6 */   \
  "vdup.32    q9,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q0,   q0,   q8\n"      /* relu6 */       \
  "vmax.f32   q1,   q1,   q8\n"      /* relu6 */       \
  "vmax.f32   q2,   q2,   q8\n"      /* relu6 */       \
  "vmax.f32   q3,   q3,   q8\n"      /* relu6 */       \
  "vmax.f32   q4,   q4,   q8\n"      /* relu6 */       \
  "vmax.f32   q5,   q5,   q8\n"      /* relu6 */       \
  "vmax.f32   q6,   q6,   q8\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q8\n"      /* relu6 */       \
  "vmin.f32   q0,   q0,   q9\n"      /* relu6 */       \
  "vmin.f32   q1,   q1,   q9\n"      /* relu6 */       \
  "vmin.f32   q2,   q2,   q9\n"      /* relu6 */       \
  "vmin.f32   q3,   q3,   q9\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q9\n"      /* relu6 */       \
  "vmin.f32   q5,   q5,   q9\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q9\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q9\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W32_v7_LEAKY_RELU                     \
  /* do relu */                                               \
  "11: \n"                                                    \
  "vmov.i32   q8, #0\n"               /* for relu */          \
  "vdup.32    q9,  %[valpha]\n"       /* leakey relu alpha */ \
  "vcge.f32   q10,    q0,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q11,    q0,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q12,    q1,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q13,    q1,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q14,    q2,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q15,    q2,    q9   \n" /* vmulq_f32 */         \
  "vbif       q0,    q11,    q10   \n"                        \
  "vbif       q1,    q13,    q12   \n"                        \
  "vbif       q2,    q15,    q14   \n"                        \
  "vcge.f32   q10,    q3,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q11,    q3,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q12,    q4,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q13,    q4,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q14,    q5,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q15,    q5,    q9   \n" /* vmulq_f32 */         \
  "vbif       q3,    q11,    q10   \n"                        \
  "vbif       q4,    q13,    q12   \n"                        \
  "vbif       q5,    q15,    q14   \n"                        \
  "vcge.f32   q10,    q6,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q11,    q6,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q12,    q7,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q13,    q7,    q9   \n" /* vmulq_f32 */         \
  "vbif       q6,    q11,    q10   \n"                        \
  "vbif       q7,    q13,    q12   \n"                        \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_INT8_F32_W32_v7_OUT                                     \
  SPARSE_INT8_F32_W32_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_FP32_W32_v7 \
      SPARSE_INT8_F32_W32_v7_RELU SPARSE_INT8_F32_W32_v7_RELU6         \
          SPARSE_INT8_F32_W32_v7_LEAKY_RELU                            \
      "mov   r0,   %[c_ptr]\n" /* store result */                      \
      "vst1.32   {d8-d11},  [r0]\n"                                    \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d12-d15},  [r0]\n"                                   \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d0-d3},  [r0]\n"                                     \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d4-d7},  [r0]\n"

#define SPARSE_INT8_F32_W16_v7_KERNEL  \
  "veor   q4,  q0,  q0\n"              \
  "pld  [%[a_ptr], #32]    \n"         \
  "veor   q5,  q1,  q1\n"              \
  "pld  [%[widx_dmap], #32]    \n"     \
  "veor   q6,  q2,  q2\n"              \
  "pld  [%[b_ptr], #32]    \n"         \
  "veor   q7,  q3,  q3\n"              \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "vld1.8  {d2-d3}, [%[b_ptr]]\n"      \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q2,    d2,  d0\n"       \
  "vmull.s8    q3,    d3,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q4,    q4,  d4\n"       \
  "vaddw.s16   q5,    q5,  d5\n"       \
  "pld  [%[b_ptr], #32]    \n"         \
  "vaddw.s16   q6,    q6,  d6\n"       \
  "vaddw.s16   q7,    q7,  d7\n"       \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_FP32_W16_v7     \
  /* write output */                               \
  "vcvt.f32.s32   q0, q4\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q1, q5\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q2, q6\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q7\n" /* cvt int32 to fp32*/ \
  "vdup.32    q4,   %[vscale]\n"                   \
  "vdup.32    q8,   %[vbias]\n"                    \
  "vdup.32    q9,   d16[0]\n"                      \
  "vdup.32    q10,  d16[0]\n"                      \
  "vdup.32    q11,  d16[0]\n"                      \
  "3:\n"                                           \
  "vmla.f32  q8,   q0,  q4\n"                      \
  "vmla.f32  q9,   q1,  q4\n"                      \
  "vmla.f32  q10,  q2,  q4\n"                      \
  "vmla.f32  q11,  q3,  q4\n"

#define SPARSE_INT8_F32_W16_v7_RELU                   \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q8,   q8,   q0\n"      /* relu */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W16_v7_RELU6                   \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W16_v7_LEAKY_RELU                    \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q2,    q8,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q9,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q9,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q6,    q10,   q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q7,    q10,   q1   \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2   \n"                         \
  "vbif       q9,    q5,    q4   \n"                         \
  "vbif       q10,   q7,    q6   \n"                         \
  "vcge.f32   q2,    q11,    q0   \n" /* vcgeq_f32 */        \
  "vmul.f32   q3,    q11,    q1   \n" /* vmulq_f32 */        \
  "vbif       q11,   q3,    q2   \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_INT8_F32_W16_v7_OUT                                     \
  SPARSE_INT8_F32_W16_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_FP32_W16_v7 \
      SPARSE_INT8_F32_W16_v7_RELU SPARSE_INT8_F32_W16_v7_RELU6         \
          SPARSE_INT8_F32_W16_v7_LEAKY_RELU                            \
      "mov   r0,   %[c_ptr]\n" /* store result */                      \
      "vst1.32   {d16-d19},  [r0]\n"                                   \
      "add  r0,  r0,   #32\n"                                          \
      "vst1.32   {d20-d23},  [r0]\n"

#define SPARSE_INT8_F32_W8_v7_KERNEL   \
  "veor   q4,  q0,  q0\n"              \
  "veor   q5,  q1,  q1\n"              \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "vld1.8  d2, [%[b_ptr]]\n"           \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q2,    d2,  d0\n"       \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "subs    %[k],   %[k],   #1\n"       \
  "vaddw.s16   q4,    q4,  d4\n"       \
  "vaddw.s16   q5,    q5,  d5\n"       \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_FP32_W8_v7      \
  /* write output */                               \
  "vcvt.f32.s32   q0, q4\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q1, q5\n" /* cvt int32 to fp32*/ \
  "vdup.32    q2,   %[vscale]\n"                   \
  "vdup.32    q6,   %[vbias]\n"                    \
  "vdup.32    q7,   d12[0]\n"                      \
  "3:\n"                                           \
  "vmla.f32  q6,   q0,  q2\n"                      \
  "vmla.f32  q7,   q1,  q2\n"

#define SPARSE_INT8_F32_W8_v7_RELU                    \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q6,   q6,   q0\n"      /* relu */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W8_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q6,   q6,   q0\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q1\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W8_v7_LEAKY_RELU                     \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q2,    q6,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q6,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q7,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q7,    q1   \n" /* vmulq_f32 */         \
  "vbif       q6,    q3,    q2   \n" /* vmulq_f32 */         \
  "vbif       q7,    q5,    q4   \n" /* vmulq_f32 */         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_INT8_F32_W8_v7_OUT                                    \
  SPARSE_INT8_F32_W8_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_FP32_W8_v7 \
      SPARSE_INT8_F32_W8_v7_RELU SPARSE_INT8_F32_W8_v7_RELU6         \
          SPARSE_INT8_F32_W8_v7_LEAKY_RELU /* store result */        \
      "vst1.32   {d12-d15},  [%[c_ptr]]\n"

#define SPARSE_INT8_F32_W4_v7_KERNEL   \
  "veor   q3,  q0,  q0\n"              \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "vdup.8     d0,    r0\n"             \
  "subs    %[k],   %[k],   #1\n"       \
  "ldr     r0, [%[b_ptr]]\n"           \
  "vdup.32    d2,   r0\n"              \
  "vmull.s8    q2,    d2,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q3,    q3,  d4\n"       \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_FP32_W4_v7      \
  /* write output */                               \
  "vcvt.f32.s32   q0, q3\n" /* cvt int32 to fp32*/ \
  "vdup.32    q1,   %[vscale]\n"                   \
  "vdup.32    q4,   %[vbias]\n"                    \
  "3:\n"                                           \
  "vmla.f32  q4,   q0,  q1\n"

#define SPARSE_INT8_F32_W4_v7_RELU                    \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q4,   q4,   q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W4_v7_RELU6                    \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q4,   q4,   q0\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_F32_W4_v7_LEAKY_RELU                     \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q2,    q4,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q4,    q1   \n" /* vmulq_f32 */         \
  "vbif       q4,    q3,    q2   \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_INT8_F32_W4_v7_OUT                                    \
  SPARSE_INT8_F32_W4_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_FP32_W4_v7 \
      SPARSE_INT8_F32_W4_v7_RELU SPARSE_INT8_F32_W4_v7_RELU6         \
          SPARSE_INT8_F32_W4_v7_LEAKY_RELU /* store result */        \
      "vst1.32   {d8-d9},  [%[c_ptr]]\n"

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
          asm volatile(SPARSE_INT8_F32_W48_v7_OUT
            : [a_ptr] "+r"(w),
              [b_ptr] "+r"(B),
              [c_ptr] "+r"(output),
              [k] "+r"(nnz),
              [widx_dmap] "+r"(dmap)
            : [vscale] "r"(vsclae),
              [vbias] "r"(vbias),
              [vflag_act] "r"(flag_act),
              [valpha] "r"(valpha)
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
              "r2",
              "cc",
              "memory");
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
            asm volatile(SPARSE_INT8_F32_W32_v7_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
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
                "r0",
                "r1",
                "r2",
                "cc",
                "memory");
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
            asm volatile(SPARSE_INT8_F32_W16_v7_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
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
                "r0",
                "r1",
                "r2",
                "cc",
                "memory");
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
            asm volatile(SPARSE_INT8_F32_W8_v7_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "r0",
                "r1",
                "r2",
                "cc",
                "memory");
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
            asm volatile(SPARSE_INT8_F32_W4_v7_OUT
              : [a_ptr] "+r"(w),
                [b_ptr] "+r"(B),
                [c_ptr] "+r"(output),
                [k] "+r"(nnz),
                [widx_dmap] "+r"(dmap)
              : [vscale] "r"(vsclae),
                [vbias] "r"(vbias),
                [vflag_act] "r"(flag_act),
                [valpha] "r"(valpha)
              : "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "r0",
                "r1",
                "r2",
                "cc",
                "memory");
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

#define SPARSE_INT8_INT8_W48_v7_KERNEL \
  "veor   q4,   q0,  q0\n"             \
  "veor   q5,   q1,  q1\n"             \
  "veor   q6,   q2,  q2\n"             \
  "veor   q7,   q3,  q3\n"             \
  "pld  [%[a_ptr], #32]    \n"         \
  "veor   q8,   q0,  q0\n"             \
  "veor   q9,   q1,  q1\n"             \
  "pld  [%[widx_dmap], #32]    \n"     \
  "veor   q10,  q2,  q2\n"             \
  "veor   q11,  q3,  q3\n"             \
  "veor   q12,  q0,  q0\n"             \
  "pld  [%[b_ptr], #64]    \n"         \
  "veor   q13,  q1,  q1\n"             \
  "veor   q14,  q2,  q2\n"             \
  "veor   q15,  q3,  q3\n"             \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "mov   r2,   %[b_ptr]\n"             \
  "vld1.8  {d2-d5}, [r2]\n"            \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q3,    d2,  d0\n"       \
  "vaddw.s16   q4,    q4,  d6\n"       \
  "vaddw.s16   q5,    q5,  d7\n"       \
  "vmull.s8    q3,    d3,  d0\n"       \
  "pld  [%[widx_dmap], #32]    \n"     \
  "vaddw.s16   q6,    q6,  d6\n"       \
  "vaddw.s16   q7,    q7,  d7\n"       \
  "vmull.s8    q3,    d4,  d0\n"       \
  "vaddw.s16   q8,    q8, d6\n"        \
  "vaddw.s16   q9,    q9, d7\n"        \
  "vmull.s8    q3,    d5,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q10,   q10, d6\n"       \
  "vaddw.s16   q11,   q11, d7\n"       \
  "pld  [%[b_ptr], #64]    \n"         \
  "add  r2,  r2,   #32\n"              \
  "vld1.8  {d2-d3}, [r2]\n"            \
  "vmull.s8    q3,    d2,  d0\n"       \
  "vaddw.s16   q12,   q12,  d6\n"      \
  "pld  [%[a_ptr], #32]    \n"         \
  "vaddw.s16   q13,   q13,  d7\n"      \
  "vmull.s8    q3,    d3,  d0\n"       \
  "vaddw.s16   q14,   q14,  d6\n"      \
  "vaddw.s16   q15,   q15,  d7\n"      \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_INT8_W48_v7      \
  /* write output */                                \
  "vdup.32    q0,   %[vscale]\n"                    \
  "vdup.32    q1,   %[vbias]\n"                     \
  "vcvt.f32.s32   q2, q4\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q5\n" /* cvt int32 to fp32*/  \
  "vdup.32    q4,   d2[0]\n"                        \
  "vdup.32    q5,   d2[0]\n"                        \
  "vmla.f32   q4,  q2,  q0\n"                       \
  "vmla.f32   q5,  q3,  q0\n"                       \
  "vcvt.f32.s32   q2, q6\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q7\n" /* cvt int32 to fp32*/  \
  "vdup.32    q6,   d2[0]\n"                        \
  "vdup.32    q7,   d2[0]\n"                        \
  "vmla.f32   q6,  q2,  q0\n"                       \
  "vmla.f32   q7,  q3,  q0\n"                       \
  "vcvt.f32.s32   q2, q8\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q9\n" /* cvt int32 to fp32*/  \
  "vdup.32    q8,   d2[0]\n"                        \
  "vdup.32    q9,   d2[0]\n"                        \
  "vmla.f32   q8,  q2,  q0\n"                       \
  "vmla.f32   q9,  q3,  q0\n"                       \
  "vcvt.f32.s32   q2, q10\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q11\n" /* cvt int32 to fp32*/ \
  "vdup.32    q10,   d2[0]\n"                       \
  "vdup.32    q11,   d2[0]\n"                       \
  "vmla.f32   q10,  q2,  q0\n"                      \
  "vmla.f32   q11,  q3,  q0\n"                      \
  "vcvt.f32.s32   q2, q12\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q13\n" /* cvt int32 to fp32*/ \
  "vdup.32    q12,   d2[0]\n"                       \
  "vdup.32    q13,   d2[0]\n"                       \
  "vmla.f32   q12,  q2,  q0\n"                      \
  "vmla.f32   q13,  q3,  q0\n"                      \
  "vcvt.f32.s32   q2, q14\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q3, q15\n" /* cvt int32 to fp32*/ \
  "vdup.32    q14,   d2[0]\n"                       \
  "vdup.32    q15,   d2[0]\n"                       \
  "vmla.f32   q14,  q2,  q0\n"                      \
  "vmla.f32   q15,  q3,  q0\n"

#define SPARSE_INT8_INT8_W48_v7_RELU                  \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q4,   q4,   q0\n"      /* relu */       \
  "vmax.f32   q5,   q5,   q0\n"      /* relu */       \
  "vmax.f32   q6,   q6,   q0\n"      /* relu */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu */       \
  "vmax.f32   q8,   q8,   q0\n"      /* relu */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu */       \
  "vmax.f32   q12,  q12,  q0\n"      /* relu */       \
  "vmax.f32   q13,  q13,  q0\n"      /* relu */       \
  "vmax.f32   q14,  q14,  q0\n"      /* relu */       \
  "vmax.f32   q15,  q15,  q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W48_v7_RELU6                  \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q4,   q4,   q0\n"      /* relu6 */       \
  "vmax.f32   q5,   q5,   q0\n"      /* relu6 */       \
  "vmax.f32   q6,   q6,   q0\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu6 */       \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmax.f32   q12,  q12,  q0\n"      /* relu6 */       \
  "vmax.f32   q13,  q13,  q0\n"      /* relu6 */       \
  "vmax.f32   q14,  q14,  q0\n"      /* relu6 */       \
  "vmax.f32   q15,  q15,  q0\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q1\n"      /* relu6 */       \
  "vmin.f32   q5,   q5,   q1\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q1\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q1\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "vmin.f32   q12,  q12,  q1\n"      /* relu6 */       \
  "vmin.f32   q13,  q13,  q1\n"      /* relu6 */       \
  "vmin.f32   q14,  q14,  q1\n"      /* relu6 */       \
  "vmin.f32   q15,  q15,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W48_v7_LEAKY_RELU                     \
  /* do relu */                                                \
  "11: \n"                                                     \
  "vmov.i32   q0, #0\n"                /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"        /* leakey relu alpha */ \
  "vcge.f32   q2,    q4,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q4,    q1     \n" /* vmulq_f32 */         \
  "vbif       q4,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q5,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q5,    q1     \n" /* vmulq_f32 */         \
  "vbif       q5,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q6,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q6,    q1     \n" /* vmulq_f32 */         \
  "vbif       q6,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q7,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q7,    q1     \n" /* vmulq_f32 */         \
  "vbif       q7,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q8,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1     \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q9,    q0     \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q9,    q1     \n" /* vmulq_f32 */         \
  "vbif       q9,    q3,    q2     \n"                         \
  "vcge.f32   q2,    q10,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q10,    q1    \n" /* vmulq_f32 */         \
  "vbif       q10,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q11,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q11,    q1    \n" /* vmulq_f32 */         \
  "vbif       q11,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q12,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q12,    q1    \n" /* vmulq_f32 */         \
  "vbif       q12,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q13,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q13,    q1    \n" /* vmulq_f32 */         \
  "vbif       q13,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q14,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q14,    q1    \n" /* vmulq_f32 */         \
  "vbif       q14,    q3,    q2    \n"                         \
  "vcge.f32   q2,    q15,    q0    \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q15,    q1    \n" /* vmulq_f32 */         \
  "vbif       q15,    q3,    q2    \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx48, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx48, and the required data is
 * MxKxKx48.
 */
#define SPARSE_INT8_INT8_W48_v7_OUT                                     \
  SPARSE_INT8_INT8_W48_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_INT8_W48_v7 \
      SPARSE_INT8_INT8_W48_v7_RELU SPARSE_INT8_INT8_W48_v7_RELU6        \
          SPARSE_INT8_INT8_W48_v7_LEAKY_RELU                            \
      "vmov.f32  q0, #-0.5\n"    /* neg offset */                       \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q2, q4, #0\n"   /* get pos mask */                     \
      "vbif.f32  q1, q0, q2\n"   /* get right offset */                 \
      "vadd.f32  q4, q1, q4\n"   /* add offset */                       \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q3, q5, #0\n"   /* get pos mask */                     \
      "vbif.f32  q2, q0, q3\n"   /* get right offset */                 \
      "vadd.f32  q5, q2, q5\n"   /* add offset */                       \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q2, q6, #0\n"   /* get pos mask */                     \
      "vbif.f32  q1, q0, q2\n"   /* get right offset */                 \
      "vadd.f32  q6, q1, q6\n"   /* add offset */                       \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q3, q7, #0\n"   /* get pos mask */                     \
      "vbif.f32  q2, q0, q3\n"   /* get right offset */                 \
      "vadd.f32  q7, q2, q7\n"   /* add offset */                       \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q2, q8, #0\n"   /* get pos mask */                     \
      "vbif.f32  q1, q0, q2\n"   /* get right offset */                 \
      "vadd.f32  q8, q1, q8\n"   /* add offset */                       \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q3, q9, #0\n"   /* get pos mask */                     \
      "vbif.f32  q2, q0, q3\n"   /* get right offset */                 \
      "vadd.f32  q9, q2, q9\n"   /* add offset */                       \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q2, q10, #0\n"  /* get pos mask */                     \
      "vbif.f32  q1, q0, q2\n"   /* get right offset */                 \
      "vadd.f32  q10, q1, q10\n" /* add offset */                       \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q3, q11, #0\n"  /* get pos mask */                     \
      "vbif.f32  q2, q0, q3\n"   /* get right offset */                 \
      "vadd.f32  q11, q2, q11\n" /* add offset */                       \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q2, q12, #0\n"  /* get pos mask */                     \
      "vbif.f32  q1, q0, q2\n"   /* get right offset */                 \
      "vadd.f32  q12, q1, q12\n" /* add offset */                       \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q3, q13, #0\n"  /* get pos mask */                     \
      "vbif.f32  q2, q0, q3\n"   /* get right offset */                 \
      "vadd.f32  q13, q2, q13\n" /* add offset */                       \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q2, q14, #0\n"  /* get pos mask */                     \
      "vbif.f32  q1, q0, q2\n"   /* get right offset */                 \
      "vadd.f32  q14, q1, q14\n" /* add offset */                       \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                       \
      "vcgt.f32  q3, q15, #0\n"  /* get pos mask */                     \
      "vbif.f32  q2, q0, q3\n"   /* get right offset */                 \
      "vadd.f32  q15, q2, q15\n" /* add offset */                       \
      "vld1.f32 {d0-d1}, [%[vmax]] \n"                                  \
      "vcge.f32 q1,  q4, q0\n"                                          \
      "vcge.f32 q2,  q5, q0\n"                                          \
      "vcge.f32 q3,  q6, q0\n"                                          \
      "vbif q4, q0, q1\n"                                               \
      "vbif q5, q0, q2\n"                                               \
      "vbif q6, q0, q3\n"                                               \
      "vcge.f32 q1,  q7, q0\n"                                          \
      "vcge.f32 q2,  q8, q0\n"                                          \
      "vcge.f32 q3,  q9, q0\n"                                          \
      "vbif q7, q0, q1\n"                                               \
      "vbif q8, q0, q2\n"                                               \
      "vbif q9, q0, q3\n"                                               \
      "vcge.f32 q1,  q10, q0\n"                                         \
      "vcge.f32 q2,  q11, q0\n"                                         \
      "vcge.f32 q3,  q12, q0\n"                                         \
      "vbif q10, q0, q1\n"                                              \
      "vbif q11, q0, q2\n"                                              \
      "vbif q12, q0, q3\n"                                              \
      "vcge.f32 q1,  q13, q0\n"                                         \
      "vcge.f32 q2,  q14, q0\n"                                         \
      "vcge.f32 q3,  q15, q0\n"                                         \
      "vbif q13, q0, q1\n"                                              \
      "vbif q14, q0, q2\n"                                              \
      "vbif q15, q0, q3\n"                                              \
      "vcvt.s32.f32   q0, q4\n"  /*     fp32->int32 */                  \
      "vcvt.s32.f32   q1, q5\n"  /*      fp32->int32 */                 \
      "vcvt.s32.f32   q2, q6\n"  /*      fp32->int32 */                 \
      "vcvt.s32.f32   q3, q7\n"  /*      fp32->int32 */                 \
      "vqmovn.s32 d8, q0\n"      /*     int32 -> int16 */               \
      "vqmovn.s32 d9, q1\n"      /*      int32 -> int16 */              \
      "vqmovn.s32 d10, q2\n"     /*      int32 -> int16 */              \
      "vqmovn.s32 d11, q3\n"     /*      int32 -> int16 */              \
      "vqmovn.s16 d12, q4\n"     /* 0, int16 -> int8 */                 \
      "vqmovn.s16 d13, q5\n"     /* 1, int16 -> int8 */                 \
      "vcvt.s32.f32   q0, q8\n"  /*     fp32->int32 */                  \
      "vcvt.s32.f32   q1, q9\n"  /*      fp32->int32 */                 \
      "vcvt.s32.f32   q2, q10\n" /*      fp32->int32 */                 \
      "vcvt.s32.f32   q3, q11\n" /*      fp32->int32 */                 \
      "vqmovn.s32 d8, q0\n"      /*     int32 -> int16 */               \
      "vqmovn.s32 d9, q1\n"      /*      int32 -> int16 */              \
      "vqmovn.s32 d10, q2\n"     /*      int32 -> int16 */              \
      "vqmovn.s32 d11, q3\n"     /*      int32 -> int16 */              \
      "vqmovn.s16 d14, q4\n"     /* 0, int16 -> int8 */                 \
      "vqmovn.s16 d15, q5\n"     /* 1, int16 -> int8 */                 \
      "vcvt.s32.f32   q0, q12\n" /*     fp32->int32 */                  \
      "vcvt.s32.f32   q1, q13\n" /*      fp32->int32 */                 \
      "vcvt.s32.f32   q2, q14\n" /*      fp32->int32 */                 \
      "vcvt.s32.f32   q3, q15\n" /*      fp32->int32 */                 \
      "vqmovn.s32 d8, q0\n"      /*     int32 -> int16 */               \
      "vqmovn.s32 d9, q1\n"      /*      int32 -> int16 */              \
      "vqmovn.s32 d10, q2\n"     /*      int32 -> int16 */              \
      "vqmovn.s32 d11, q3\n"     /*      int32 -> int16 */              \
      "vqmovn.s16 d16, q4\n"     /* 0, int16 -> int8 */                 \
      "vqmovn.s16 d17, q5\n"     /* 1, int16 -> int8 */                 \
      "mov   r0,   %[c_ptr]\n"   /* store result */                     \
      "vst1.32   {d12-d13},  [r0]\n"                                    \
      "add  r0,  r0,   #16\n"                                           \
      "vst1.32   {d14-d15},  [r0]\n"                                    \
      "add  r0,  r0,   #16\n"                                           \
      "vst1.32   {d16-d17},  [r0]\n"

#define SPARSE_INT8_INT8_W32_v7_KERNEL \
  "veor   q8,   q0,  q0\n"             \
  "veor   q9,   q1,  q1\n"             \
  "pld  [%[a_ptr], #32]    \n"         \
  "veor   q10,  q2,  q2\n"             \
  "veor   q11,  q3,  q3\n"             \
  "pld  [%[widx_dmap], #32]    \n"     \
  "veor   q12,  q4,  q4\n"             \
  "veor   q13,  q5,  q5\n"             \
  "pld  [%[b_ptr], #64]    \n"         \
  "veor   q14,  q6,  q6\n"             \
  "veor   q15,  q7,  q7\n"             \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "vld1.8  {d2-d5}, [%[b_ptr]]\n"      \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q3,    d2,  d0\n"       \
  "vmull.s8    q4,    d3,  d0\n"       \
  "pld  [%[widx_dmap], #32]    \n"     \
  "vmull.s8    q5,    d4,  d0\n"       \
  "vmull.s8    q6,    d5,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q8,     q8,  d6\n"      \
  "vaddw.s16   q9,     q9,  d7\n"      \
  "pld  [%[a_ptr], #32]    \n"         \
  "vaddw.s16   q10,    q10,  d8\n"     \
  "vaddw.s16   q11,    q11,  d9\n"     \
  "vaddw.s16   q12,    q12,  d10\n"    \
  "pld  [%[b_ptr], #64]    \n"         \
  "vaddw.s16   q13,    q13,  d11\n"    \
  "vaddw.s16   q14,    q14,  d12\n"    \
  "vaddw.s16   q15,    q15,  d13\n"    \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_INT8_W32_v7       \
  /* write output */                                 \
  "vcvt.f32.s32   q0, q8\n"  /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q1, q9\n"  /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q2, q10\n" /* cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q11\n" /* cvt int32 to fp32*/  \
  "vdup.32    q8,   %[vscale]\n"                     \
  "vdup.32    q4,   %[vbias]\n"                      \
  "vdup.32    q5,   d8[0]\n"                         \
  "vdup.32    q6,   d8[0]\n"                         \
  "vdup.32    q7,   d8[0]\n"                         \
  "3:\n"                                             \
  "vmla.f32  q4,  q0,  q8\n"                         \
  "vmla.f32  q5,  q1,  q8\n"                         \
  "vmla.f32  q6,  q2,  q8\n"                         \
  "vmla.f32  q7,  q3,  q8\n"                         \
  "4:\n"                                             \
  "vcvt.f32.s32   q8, q12\n"  /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q9, q13\n"  /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q10, q14\n" /* cvt int32 to fp32*/ \
  "vcvt.f32.s32   q11, q15\n" /* cvt int32 to fp32*/ \
  "vdup.32    q12,   %[vscale]\n"                    \
  "vdup.32    q0,   %[vbias]\n"                      \
  "vdup.32    q1,   d0[0]\n"                         \
  "vdup.32    q2,   d0[0]\n"                         \
  "vdup.32    q3,   d0[0]\n"                         \
  "6:\n"                                             \
  "vmla.f32  q0,  q8,  q12\n"                        \
  "vmla.f32  q1,  q9,  q12\n"                        \
  "vmla.f32  q2,  q10,  q12\n"                       \
  "vmla.f32  q3,  q11,  q12\n"

#define SPARSE_INT8_INT8_W32_v7_RELU                  \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q8, #0\n"              /* for relu */   \
  "vmax.f32   q0,   q0,   q8\n"      /* relu */       \
  "vmax.f32   q1,   q1,   q8\n"      /* relu */       \
  "vmax.f32   q2,   q2,   q8\n"      /* relu */       \
  "vmax.f32   q3,   q3,   q8\n"      /* relu */       \
  "vmax.f32   q4,   q4,   q8\n"      /* relu */       \
  "vmax.f32   q5,   q5,   q8\n"      /* relu */       \
  "vmax.f32   q6,   q6,   q8\n"      /* relu */       \
  "vmax.f32   q7,   q7,   q8\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W32_v7_RELU6                  \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q8,   #0\n"            /* for relu6 */   \
  "vdup.32    q9,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q0,   q0,   q8\n"      /* relu6 */       \
  "vmax.f32   q1,   q1,   q8\n"      /* relu6 */       \
  "vmax.f32   q2,   q2,   q8\n"      /* relu6 */       \
  "vmax.f32   q3,   q3,   q8\n"      /* relu6 */       \
  "vmax.f32   q4,   q4,   q8\n"      /* relu6 */       \
  "vmax.f32   q5,   q5,   q8\n"      /* relu6 */       \
  "vmax.f32   q6,   q6,   q8\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q8\n"      /* relu6 */       \
  "vmin.f32   q0,   q0,   q9\n"      /* relu6 */       \
  "vmin.f32   q1,   q1,   q9\n"      /* relu6 */       \
  "vmin.f32   q2,   q2,   q9\n"      /* relu6 */       \
  "vmin.f32   q3,   q3,   q9\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q9\n"      /* relu6 */       \
  "vmin.f32   q5,   q5,   q9\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q9\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q9\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W32_v7_LEAKY_RELU                    \
  /* do relu */                                               \
  "11: \n"                                                    \
  "vmov.i32   q8, #0\n"               /* for relu */          \
  "vdup.32    q9,  %[valpha]\n"       /* leakey relu alpha */ \
  "vcge.f32   q10,    q0,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q11,    q0,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q12,    q1,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q13,    q1,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q14,    q2,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q15,    q2,    q9   \n" /* vmulq_f32 */         \
  "vbif       q0,    q11,    q10   \n"                        \
  "vbif       q1,    q13,    q12   \n"                        \
  "vbif       q2,    q15,    q14   \n"                        \
  "vcge.f32   q10,    q3,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q11,    q3,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q12,    q4,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q13,    q4,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q14,    q5,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q15,    q5,    q9   \n" /* vmulq_f32 */         \
  "vbif       q3,    q11,    q10   \n"                        \
  "vbif       q4,    q13,    q12   \n"                        \
  "vbif       q5,    q15,    q14   \n"                        \
  "vcge.f32   q10,    q6,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q11,    q6,    q9   \n" /* vmulq_f32 */         \
  "vcge.f32   q12,    q7,    q8   \n" /* vcgeq_f32 */         \
  "vmul.f32   q13,    q7,    q9   \n" /* vmulq_f32 */         \
  "vbif       q6,    q11,    q10   \n"                        \
  "vbif       q7,    q13,    q12   \n"                        \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx32, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx32, and the required data is
 * MxKxKx32.
 */
#define SPARSE_INT8_INT8_W32_v7_OUT                                            \
  SPARSE_INT8_INT8_W32_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_INT8_W32_v7        \
      SPARSE_INT8_INT8_W32_v7_RELU SPARSE_INT8_INT8_W32_v7_RELU6               \
          SPARSE_INT8_INT8_W32_v7_LEAKY_RELU                                   \
      "vmov.f32  q8, #-0.5\n"    /* neg offset */                              \
      "vmov.f32  q10, #0.5\n"    /* pos offset */                              \
      "vmov.f32  q11, #0.5\n"    /* pos offset */                              \
      "vmov.f32  q12, #0.5\n"    /* pos offset */                              \
      "vmov.f32  q13, #0.5\n"    /* pos offset */                              \
      "vcgt.f32  q14, q0, #0\n"  /* get pos mask */                            \
      "vcgt.f32  q15, q1, #0\n"  /* get pos mask */                            \
      "vbif.f32  q10, q8, q14\n" /* get right offset */                        \
      "vbif.f32  q11, q8, q15\n" /* get right offset */                        \
      "vcgt.f32  q14, q2, #0\n"  /* get pos mask */                            \
      "vcgt.f32  q15, q3, #0\n"  /* get pos mask */                            \
      "vbif.f32  q12, q8, q14\n" /* get right offset */                        \
      "vbif.f32  q13, q8, q15\n" /* get right offset */                        \
      "vadd.f32  q0, q10, q0\n"  /* add offset */                              \
      "vadd.f32  q1, q11, q1\n"  /* add offset */                              \
      "vadd.f32  q2, q12, q2\n"  /* add offset */                              \
      "vadd.f32  q3, q13, q3\n"  /* add offset */                              \
      "vmov.f32  q10, #0.5\n"    /* pos offset */                              \
      "vmov.f32  q11, #0.5\n"    /* pos offset */                              \
      "vmov.f32  q12, #0.5\n"    /* pos offset */                              \
      "vmov.f32  q13, #0.5\n"    /* pos offset */                              \
      "vcgt.f32  q14, q4, #0\n"  /* get pos mask */                            \
      "vcgt.f32  q15, q5, #0\n"  /* get pos mask */                            \
      "vbif.f32  q10, q8, q14\n" /* get right offset */                        \
      "vbif.f32  q11, q8, q15\n" /* get right offset */                        \
      "vcgt.f32  q14, q6, #0\n"  /* get pos mask */                            \
      "vcgt.f32  q15, q7, #0\n"  /* get pos mask */                            \
      "vbif.f32  q12, q8, q14\n" /* get right offset */                        \
      "vbif.f32  q13, q8, q15\n" /* get right offset */                        \
      "vadd.f32  q4, q10, q4\n"  /*      add offset */                         \
      "vadd.f32  q5, q11, q5\n"  /*      add offset */                         \
      "vadd.f32  q6, q12, q6\n"  /*      add offset */                         \
      "vadd.f32  q7, q13, q7\n"  /*      add offset */                         \
      "vld1.f32 {d16-d17}, [%[vmax]] \n"                                       \
      "vcge.f32 q9,  q0, q8\n"                       /* @ q8 >= -127 \n */     \
      "vcge.f32 q10, q1, q8\n"                       /* @ q9 >= -127 \n */     \
      "vcge.f32 q11, q2, q8\n"                       /* @ q0 >= -127 \n */     \
      "vcge.f32 q12, q3, q8\n"                       /* @ q1 >= -127 \n */     \
      "vcge.f32 q13, q4, q8\n"                       /* @ q2 >= -127 \n */     \
      "vcge.f32 q14, q5, q8\n"                       /* @ q3 >= -127 \n */     \
      "vcge.f32 q15, q6, q8\n" /* @ q4 >= -127 \n */ /* choose data */         \
      "vbif q0, q8, q9\n"                            /* @ choose */            \
      "vcge.f32 q9,  q7, q8\n"                       /* @ q5 >= -127 \n */     \
      "vbif q1, q8, q10\n"                           /* @ choose */            \
      "vbif q2, q8, q11\n"                           /* @ choose */            \
      "vbif q3, q8, q12\n"                           /* @ choose */            \
      "vbif q4, q8, q13\n"                           /* @ choose */            \
      "vbif q5, q8, q14\n"                           /* @ choose */            \
      "vbif q6, q8, q15\n"                           /* @ choose */            \
      "vbif q7, q8, q9\n"                            /* @ choose */            \
      "vcvt.s32.f32   q8, q0\n"                      /*     fp32->int32 */     \
      "vcvt.s32.f32   q9, q1\n"                      /*      fp32->int32 */    \
      "vcvt.s32.f32   q10, q2\n"                     /*      fp32->int32 */    \
      "vcvt.s32.f32   q11, q3\n"                     /*      fp32->int32 */    \
      "vcvt.s32.f32   q12, q4\n"                     /*      fp32->int32 */    \
      "vcvt.s32.f32   q13, q5\n"                     /*      fp32->int32 */    \
      "vcvt.s32.f32   q14, q6\n"                     /*      fp32->int32 */    \
      "vcvt.s32.f32   q15, q7\n"                     /*      fp32->int32 */    \
      "vqmovn.s32 d0, q8\n"                          /*     int32 -> int16 */  \
      "vqmovn.s32 d1, q9\n"                          /*      int32 -> int16 */ \
      "vqmovn.s32 d2, q10\n"                         /*      int32 -> int16 */ \
      "vqmovn.s32 d3, q11\n"                         /*      int32 -> int16 */ \
      "vqmovn.s32 d4, q12\n"                         /*     int32 -> int16 */  \
      "vqmovn.s32 d5, q13\n"                         /*      int32 -> int16 */ \
      "vqmovn.s32 d6, q14\n"                         /*      int32 -> int16 */ \
      "vqmovn.s32 d7, q15\n"                         /*      int32 -> int16 */ \
      "vqmovn.s16 d8, q0\n"                          /* 0, int16 -> int8 */    \
      "vqmovn.s16 d9, q1\n"                          /* 1, int16 -> int8 */    \
      "vqmovn.s16 d10, q2\n"                         /* 2, int16 -> int8 */    \
      "vqmovn.s16 d11, q3\n"                         /* 3, int16 -> int8 */    \
      "mov   r0,   %[c_ptr]\n"                       /* store result */        \
      "vst1.32   {d10-d11},  [r0]\n"                                           \
      "add  r0,  r0,   #16\n"                                                  \
      "vst1.32   {d8-d9},  [r0]\n"

#define SPARSE_INT8_INT8_W16_v7_KERNEL \
  "veor   q4,  q0,  q0\n"              \
  "pld  [%[a_ptr], #32]    \n"         \
  "veor   q5,  q1,  q1\n"              \
  "pld  [%[widx_dmap], #32]    \n"     \
  "veor   q6,  q2,  q2\n"              \
  "pld  [%[b_ptr], #32]    \n"         \
  "veor   q7,  q3,  q3\n"              \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "vld1.8  {d2-d3}, [%[b_ptr]]\n"      \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q2,    d2,  d0\n"       \
  "vmull.s8    q3,    d3,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q4,    q4,  d4\n"       \
  "vaddw.s16   q5,    q5,  d5\n"       \
  "pld  [%[b_ptr], #32]    \n"         \
  "vaddw.s16   q6,    q6,  d6\n"       \
  "vaddw.s16   q7,    q7,  d7\n"       \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_INT8_W16_v7          \
  /* write output */                                    \
  "vcvt.f32.s32   q0, q4\n" /*     cvt int32 to fp32*/  \
  "vcvt.f32.s32   q1, q5\n" /*      cvt int32 to fp32*/ \
  "vcvt.f32.s32   q2, q6\n" /*     cvt int32 to fp32*/  \
  "vcvt.f32.s32   q3, q7\n" /*      cvt int32 to fp32*/ \
  "vdup.32    q4,   %[vscale]\n"                        \
  "vdup.32    q8,   %[vbias]\n"                         \
  "vdup.32    q9,   d16[0]\n"                           \
  "vdup.32    q10,  d16[0]\n"                           \
  "vdup.32    q11,  d16[0]\n"                           \
  "3:\n"                                                \
  "vmla.f32  q8,   q0,  q4\n"                           \
  "vmla.f32  q9,   q1,  q4\n"                           \
  "vmla.f32  q10,  q2,  q4\n"                           \
  "vmla.f32  q11,  q3,  q4\n"

#define SPARSE_INT8_INT8_W16_v7_RELU                  \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q8,   q8,   q0\n"      /* relu */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W16_v7_RELU6                  \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q8,   q8,   q0\n"      /* relu6 */       \
  "vmax.f32   q9,   q9,   q0\n"      /* relu6 */       \
  "vmax.f32   q10,  q10,  q0\n"      /* relu6 */       \
  "vmax.f32   q11,  q11,  q0\n"      /* relu6 */       \
  "vmin.f32   q8,   q8,   q1\n"      /* relu6 */       \
  "vmin.f32   q9,   q9,   q1\n"      /* relu6 */       \
  "vmin.f32   q10,  q10,  q1\n"      /* relu6 */       \
  "vmin.f32   q11,  q11,  q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W16_v7_LEAKY_RELU                   \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q2,    q8,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q8,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q9,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q9,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q6,    q10,   q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q7,    q10,   q1   \n" /* vmulq_f32 */         \
  "vbif       q8,    q3,    q2   \n"                         \
  "vbif       q9,    q5,    q4   \n"                         \
  "vbif       q10,   q7,    q6   \n"                         \
  "vcge.f32   q2,    q11,    q0   \n" /* vcgeq_f32 */        \
  "vmul.f32   q3,    q11,    q1   \n" /* vmulq_f32 */        \
  "vbif       q11,   q3,    q2   \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx16, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx16, and the required data is
 * MxKxKx16.
 */
#define SPARSE_INT8_INT8_W16_v7_OUT                                          \
  SPARSE_INT8_INT8_W16_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_INT8_W16_v7      \
      SPARSE_INT8_INT8_W16_v7_RELU SPARSE_INT8_INT8_W16_v7_RELU6             \
          SPARSE_INT8_INT8_W16_v7_LEAKY_RELU                                 \
      "vmov.f32  q0, #-0.5\n"    /* neg offset */                            \
      "vmov.f32  q1, #0.5\n"     /* pos offset */                            \
      "vmov.f32  q2, #0.5\n"     /* pos offset */                            \
      "vmov.f32  q3, #0.5\n"     /* pos offset */                            \
      "vmov.f32  q4, #0.5\n"     /* pos offset */                            \
      "vcgt.f32  q5, q8, #0\n"   /* get pos mask */                          \
      "vcgt.f32  q6, q9, #0\n"   /* get pos mask */                          \
      "vbif.f32  q1, q0, q5\n"   /* get right offset */                      \
      "vbif.f32  q2, q0, q6\n"   /* get right offset */                      \
      "vcgt.f32  q12, q10, #0\n" /* get pos mask */                          \
      "vcgt.f32  q13, q11, #0\n" /* get pos mask */                          \
      "vbif.f32  q3, q0, q12\n"  /* get right offset */                      \
      "vbif.f32  q4, q0, q13\n"  /* get right offset */                      \
      "vadd.f32  q8, q1, q8\n"   /*     add offset */                        \
      "vadd.f32  q9, q2, q9\n"   /*      add offset */                       \
      "vadd.f32  q10, q3, q10\n" /*      add offset */                       \
      "vadd.f32  q11, q4, q11\n" /*      add offset */                       \
      "vld1.f32 {d14-d15}, [%[vmax]] \n"                                     \
      "vcge.f32 q0,  q8,  q7\n"                    /* @ q8 >= -127 \n */     \
      "vcge.f32 q1,  q9,  q7\n"                    /* @ q9 >= -127 \n */     \
      "vcge.f32 q2,  q10, q7\n"                    /* @ q0 >= -127 \n */     \
      "vcge.f32 q3,  q11, q7\n"                    /* @ q1 >= -127 \n */     \
      "vbif q8, q7, q0\n"                          /* @ choose */            \
      "vbif q9, q7, q1\n"                          /* @ choose */            \
      "vbif q10, q7, q2\n"                         /* @ choose */            \
      "vbif q11, q7, q3\n"                         /* @ choose */            \
      "vcvt.s32.f32   q0, q8\n"                    /*     fp32->int32 */     \
      "vcvt.s32.f32   q1, q9\n"                    /*      fp32->int32 */    \
      "vcvt.s32.f32   q2, q10\n"                   /*      fp32->int32 */    \
      "vcvt.s32.f32   q3, q11\n"                   /*      fp32->int32 */    \
      "vqmovn.s32 d8, q0\n"                        /*     int32 -> int16 */  \
      "vqmovn.s32 d9, q1\n"                        /*      int32 -> int16 */ \
      "vqmovn.s32 d10, q2\n"                       /*      int32 -> int16 */ \
      "vqmovn.s32 d11, q3\n"                       /*      int32 -> int16 */ \
      "vqmovn.s16 d0, q4\n"                        /* 0, int16 -> int8 */    \
      "vqmovn.s16 d1, q5\n" /* 1, int16 -> int8 */ /* store result */        \
      "vst1.32   {d0-d1},  [%[c_ptr]]\n"

#define SPARSE_INT8_INT8_W8_v7_KERNEL  \
  "veor   q4,  q0,  q0\n"              \
  "veor   q5,  q1,  q1\n"              \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "subs    %[k],   %[k],   #1\n"       \
  "vld1.8  d2, [%[b_ptr]]\n"           \
  "vdup.8    d0,   r0\n"               \
  "vmull.s8    q2,    d2,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q4,    q4,  d4\n"       \
  "vaddw.s16   q5,    q5,  d5\n"       \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_INT8_W8_v7           \
  /* write output */                                    \
  "vcvt.f32.s32   q0, q4\n" /*     cvt int32 to fp32*/  \
  "vcvt.f32.s32   q1, q5\n" /*      cvt int32 to fp32*/ \
  "vdup.32    q2,   %[vscale]\n"                        \
  "vdup.32    q6,   %[vbias]\n"                         \
  "vdup.32    q7,   d12[0]\n"                           \
  "3:\n"                                                \
  "vmla.f32  q6,   q0,  q2\n"                           \
  "vmla.f32  q7,   q1,  q2\n"

#define SPARSE_INT8_INT8_W8_v7_RELU                   \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q6,   q6,   q0\n"      /* relu */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W8_v7_RELU6                   \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q6,   q6,   q0\n"      /* relu6 */       \
  "vmax.f32   q7,   q7,   q0\n"      /* relu6 */       \
  "vmin.f32   q6,   q6,   q1\n"      /* relu6 */       \
  "vmin.f32   q7,   q7,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W8_v7_LEAKY_RELU                    \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q2,    q6,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q6,    q1   \n" /* vmulq_f32 */         \
  "vcge.f32   q4,    q7,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q5,    q7,    q1   \n" /* vmulq_f32 */         \
  "vbif       q6,    q3,    q2   \n"                         \
  "vbif       q7,    q5,    q4   \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx8, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx8, and the required data is
 * MxKxKx8.
 */
#define SPARSE_INT8_INT8_W8_v7_OUT                                           \
  SPARSE_INT8_INT8_W8_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_INT8_W8_v7        \
      SPARSE_INT8_INT8_W8_v7_RELU SPARSE_INT8_INT8_W8_v7_RELU6               \
          SPARSE_INT8_INT8_W8_v7_LEAKY_RELU                                  \
      "vmov.f32  q8, #-0.5\n"  /* neg offset */                              \
      "vmov.f32  q0, #0.5\n"   /* pos offset */                              \
      "vmov.f32  q1, #0.5\n"   /* pos offset */                              \
      "vmov.f32  q2, #0.5\n"   /* pos offset */                              \
      "vmov.f32  q3, #0.5\n"   /* pos offset */                              \
      "vcgt.f32  q4, q6, #0\n" /* get pos mask */                            \
      "vcgt.f32  q5, q7, #0\n" /* get pos mask */                            \
      "vbif.f32  q0, q8, q4\n" /* get right offset */                        \
      "vbif.f32  q1, q8, q5\n" /* get right offset */                        \
      "vadd.f32  q6, q0, q6\n" /*     add offset */                          \
      "vadd.f32  q7, q1, q7\n" /*      add offset */                         \
      "vld1.f32 {d0-d1}, [%[vmax]] \n"                                       \
      "vcge.f32 q1,  q6,  q0\n"                    /* @ q8 >= -127 \n */     \
      "vcge.f32 q2,  q7,  q0\n"                    /* @ q9 >= -127 \n */     \
      "vbif q6, q0, q1\n"                          /* @ choose */            \
      "vbif q7, q0, q2\n"                          /* @ choose */            \
      "vcvt.s32.f32   q0, q6\n"                    /*     fp32->int32 */     \
      "vcvt.s32.f32   q1, q7\n"                    /*      fp32->int32 */    \
      "vqmovn.s32 d8, q0\n"                        /*     int32 -> int16 */  \
      "vqmovn.s32 d9, q1\n"                        /*      int32 -> int16 */ \
      "vqmovn.s16 d0, q4\n" /* 0, int16 -> int8 */ /* store result */        \
      "vst1.32   {d0},  [%[c_ptr]]\n"

#define SPARSE_INT8_INT8_W4_v7_KERNEL  \
  "veor   q3,  q0,  q0\n"              \
  "cmp    %[k],    #0\n"               \
  "beq    1f\n" /* main loop*/         \
  "0:\n"                               \
  "ldrsb   r0, [%[a_ptr]], #1\n"       \
  "vdup.8     d0,    r0\n"             \
  "subs    %[k],   %[k],   #1\n"       \
  "ldr     r0, [%[b_ptr]]\n"           \
  "vdup.32    d2,   r0\n"              \
  "vmull.s8    q2,    d2,  d0\n"       \
  "ldr     r1, [%[widx_dmap]],   #4\n" \
  "add   %[b_ptr],  %[b_ptr], r1\n"    \
  "vaddw.s16   q3,    q3,  d4\n"       \
  "bne     0b\n"                       \
  "1:\n"

#define SPARSE_INT8_TRANS_INT32_TO_INT8_W4_v7          \
  /* write output */                                   \
  "vcvt.f32.s32   q0, q3\n" /*     cvt int32 to fp32*/ \
  "vdup.32    q1,   %[vscale]\n"                       \
  "vdup.32    q4,   %[vbias]\n"                        \
  "3:\n"                                               \
  "vmla.f32  q4,   q0,  q1\n"

#define SPARSE_INT8_INT8_W4_v7_RELU                   \
  /* do relu */                                       \
  "cmp    %[vflag_act],    #0\n"     /* skip relu */  \
  "beq   9f                     \n"  /* no act end */ \
  "cmp    %[vflag_act],    #1\n"     /* skip relu */  \
  "bne   10f                     \n" /* other act */  \
  "vmov.i32   q0, #0\n"              /* for relu */   \
  "vmax.f32   q4,   q4,   q0\n"      /* relu */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W4_v7_RELU6                   \
  /* do relu6 */                                       \
  "10: \n"                                             \
  "cmp   %[vflag_act],  #2       \n" /* check relu6 */ \
  "bne   11f                     \n" /* no act end */  \
  "vmov.i32   q0,   #0\n"            /* for relu6 */   \
  "vdup.32    q1,   %[valpha]\n"     /* relu6 alpha */ \
  "vmax.f32   q4,   q4,   q0\n"      /* relu6 */       \
  "vmin.f32   q4,   q4,   q1\n"      /* relu6 */       \
  "b      9f                    \n"  /* relu end */

#define SPARSE_INT8_INT8_W4_v7_LEAKY_RELU                    \
  /* do relu */                                              \
  "11: \n"                                                   \
  "vmov.i32   q0, #0\n"              /* for relu */          \
  "vdup.32    q1,  %[valpha]\n"      /* leakey relu alpha */ \
  "vcge.f32   q2,    q4,    q0   \n" /* vcgeq_f32 */         \
  "vmul.f32   q3,    q4,    q1   \n" /* vmulq_f32 */         \
  "vbif       q4,    q3,    q2   \n"                         \
  "9:\n"

/**
 * The data block size for sparse matrix calculation is Mx4, that is, the
 * parameter
 * matrix size is MxK, the activation matrix is Kx4, and the required data is
 * MxKxKx4.
 */
#define SPARSE_INT8_INT8_W4_v7_OUT                                          \
  SPARSE_INT8_INT8_W4_v7_KERNEL SPARSE_INT8_TRANS_INT32_TO_INT8_W4_v7       \
      SPARSE_INT8_INT8_W4_v7_RELU SPARSE_INT8_INT8_W4_v7_RELU6              \
          SPARSE_INT8_INT8_W4_v7_LEAKY_RELU                                 \
      "vmov.f32  q1, #-0.5\n"  /* neg offset */                             \
      "vmov.f32  q0, #0.5\n"   /* pos offset */                             \
      "vcgt.f32  q2, q4, #0\n" /* get pos mask */                           \
      "vbif.f32  q0, q1, q2\n" /* get right offset */                       \
      "vadd.f32  q4, q0, q4\n" /*     add offset */                         \
      "vld1.f32 {d0-d1}, [%[vmax]] \n"                                      \
      "vcge.f32 q1,  q4,  q0\n"                    /* @ q8 >= -127 \n */    \
      "vbif q4, q0, q1\n"                          /* @ choose */           \
      "vcvt.s32.f32   q0, q4\n"                    /*     fp32->int32 */    \
      "vqmovn.s32 d2, q0\n"                        /*     int32 -> int16 */ \
      "vqmovn.s16 d0, q1\n" /* 0, int16 -> int8 */ /* store result */       \
      "vst1.32   d0[0],  [%[c_ptr]]\n"

/**
 * \brief Sparse calculation implementation of 1x1 convolution, both input and
 * output are int8.
 * Sparse matrix multiplication is calculated in blocks, the block size is Mx48,
 * that is,
 * the parameter matrix is MxK, and the activation matrix is Kx48; when N is
 * less than 48,
 * it is calculated in blocks of Mx32, Mx16, Mx8, and Mx4 in turn;
 * \brief the input type is int8, and the output type is also int8.
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
          asm volatile(SPARSE_INT8_INT8_W48_v7_OUT
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
              "r2",
              "cc",
              "memory");
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
            asm volatile(SPARSE_INT8_INT8_W32_v7_OUT
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
                "r0",
                "r1",
                "cc",
                "memory");
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
            asm volatile(SPARSE_INT8_INT8_W16_v7_OUT
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
                "r0",
                "r1",
                "cc",
                "memory");
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
            asm volatile(SPARSE_INT8_INT8_W8_v7_OUT
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
              : "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "r0",
                "r1",
                "cc",
                "memory");
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
            asm volatile(SPARSE_INT8_INT8_W4_v7_OUT
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
              : "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "r0",
                "r1",
                "cc",
                "memory");
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

#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
