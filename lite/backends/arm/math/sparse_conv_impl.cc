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
  bool has_bias = bias != nullptr;
  size_t mc = N * sizeof(float);
  size_t nc = M;
  size_t output_stride = N * sizeof(float);
  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while
    SPARSE_LIKELY(mc >= 32 * sizeof(float)) {
      const float* w = A;
      const float* b = bias;
      float valpha = alpha;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      float32x4_t vw = vld1q_dup_f32(w);
      w += 1;
      float32x4_t vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
      b += 1;
      intptr_t diff = *dmap++;
      float32x4_t vi0123 = vld1q_f32(B);
      float32x4_t vi4567 = vld1q_f32(B + 4);
      float32x4_t vi89AB = vld1q_f32(B + 8);
      float32x4_t viCDEF = vld1q_f32(B + 12);
      float32x4_t viGHIJ = vld1q_f32(B + 16);
      float32x4_t viKLMN = vld1q_f32(B + 20);
      float32x4_t viOPQR = vld1q_f32(B + 24);
      float32x4_t viSTUV = vld1q_f32(B + 28);
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vb;
        float32x4_t vacc4567 = vb;
        float32x4_t vacc89AB = vb;
        float32x4_t vaccCDEF = vb;
        float32x4_t vaccGHIJ = vb;
        float32x4_t vaccKLMN = vb;
        float32x4_t vaccOPQR = vb;
        float32x4_t vaccSTUV = vb;
        vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
        b += 1;
        __builtin_prefetch(b + 32);
        if
          SPARSE_LIKELY(nnz != 0) {
            do {
              vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
              vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vw);
              vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vw);
              vaccGHIJ = vmlaq_f32(vaccGHIJ, viGHIJ, vw);
              vaccKLMN = vmlaq_f32(vaccKLMN, viKLMN, vw);
              vaccOPQR = vmlaq_f32(vaccOPQR, viOPQR, vw);
              vaccSTUV = vmlaq_f32(vaccSTUV, viSTUV, vw);
              B = (const float*)((uintptr_t)B + (uintptr_t)diff);
              __builtin_prefetch(B + 16);
              __builtin_prefetch(B + 32);
              diff = *dmap++;
              vw = vld1q_dup_f32(w);
              w += 1;
              __builtin_prefetch(w + 32);
              vi0123 = vld1q_f32(B);
              vi4567 = vld1q_f32(B + 4);
              vi89AB = vld1q_f32(B + 8);
              viCDEF = vld1q_f32(B + 12);
              viGHIJ = vld1q_f32(B + 16);
              viKLMN = vld1q_f32(B + 20);
              viOPQR = vld1q_f32(B + 24);
              viSTUV = vld1q_f32(B + 28);
            } while (--nnz != 0);
          }
        if (flag_act == 1) {  // relu
          float32x4_t vzero = vdupq_n_f32(0);
          vacc0123 = vmaxq_f32(vacc0123, vzero);
          vacc4567 = vmaxq_f32(vacc4567, vzero);
          vacc89AB = vmaxq_f32(vacc89AB, vzero);
          vaccCDEF = vmaxq_f32(vaccCDEF, vzero);
          vaccGHIJ = vmaxq_f32(vaccGHIJ, vzero);
          vaccKLMN = vmaxq_f32(vaccKLMN, vzero);
          vaccOPQR = vmaxq_f32(vaccOPQR, vzero);
          vaccSTUV = vmaxq_f32(vaccSTUV, vzero);
        } else if (flag_act == 2) {  // relu6
          float32x4_t vzero = vdupq_n_f32(0);
          float32x4_t aph = vdupq_n_f32(valpha);
          vacc0123 = vmaxq_f32(vacc0123, vzero);
          vacc4567 = vmaxq_f32(vacc4567, vzero);
          vacc89AB = vmaxq_f32(vacc89AB, vzero);
          vaccCDEF = vmaxq_f32(vaccCDEF, vzero);
          vaccGHIJ = vmaxq_f32(vaccGHIJ, vzero);
          vaccKLMN = vmaxq_f32(vaccKLMN, vzero);
          vaccOPQR = vmaxq_f32(vaccOPQR, vzero);
          vaccSTUV = vmaxq_f32(vaccSTUV, vzero);
          vacc0123 = vminq_f32(vacc0123, aph);
          vacc4567 = vminq_f32(vacc4567, aph);
          vacc89AB = vminq_f32(vacc89AB, aph);
          vaccCDEF = vminq_f32(vaccCDEF, aph);
          vaccGHIJ = vminq_f32(vaccGHIJ, aph);
          vaccKLMN = vminq_f32(vaccKLMN, aph);
          vaccOPQR = vminq_f32(vaccOPQR, aph);
          vaccSTUV = vminq_f32(vaccSTUV, aph);
        } else if (flag_act != 0) {  // leaky_relu
          float32x4_t vzero = vdupq_n_f32(0);
          float32x4_t aph = vdupq_n_f32(valpha);
          uint32x4_t vflag0123 = vcgeq_f32(vacc0123, vzero);
          uint32x4_t vflag4567 = vcgeq_f32(vacc4567, vzero);
          uint32x4_t vflag89AB = vcgeq_f32(vacc89AB, vzero);
          uint32x4_t vflagCDEF = vcgeq_f32(vaccCDEF, vzero);
          uint32x4_t vflagGHIJ = vcgeq_f32(vaccGHIJ, vzero);
          uint32x4_t vflagKLMN = vcgeq_f32(vaccKLMN, vzero);
          uint32x4_t vflagOPQR = vcgeq_f32(vaccOPQR, vzero);
          uint32x4_t vflagSTUV = vcgeq_f32(vaccSTUV, vzero);
          float32x4_t v0123 = vmulq_f32(vacc0123, aph);
          float32x4_t v4567 = vmulq_f32(vacc4567, aph);
          float32x4_t v89AB = vmulq_f32(vacc89AB, aph);
          float32x4_t vCDEF = vmulq_f32(vaccCDEF, aph);
          float32x4_t vGHIJ = vmulq_f32(vaccGHIJ, aph);
          float32x4_t vKLMN = vmulq_f32(vaccKLMN, aph);
          float32x4_t vOPQR = vmulq_f32(vaccOPQR, aph);
          float32x4_t vSTUV = vmulq_f32(vaccSTUV, aph);
          vacc0123 = vbslq_f32(vflag0123, vacc0123, v0123);
          vacc4567 = vbslq_f32(vflag4567, vacc4567, v4567);
          vacc89AB = vbslq_f32(vflag89AB, vacc89AB, v89AB);
          vaccCDEF = vbslq_f32(vflagCDEF, vaccCDEF, vCDEF);
          vaccGHIJ = vbslq_f32(vflagGHIJ, vaccGHIJ, vGHIJ);
          vaccKLMN = vbslq_f32(vflagKLMN, vaccKLMN, vKLMN);
          vaccOPQR = vbslq_f32(vflagOPQR, vaccOPQR, vOPQR);
          vaccSTUV = vbslq_f32(vflagSTUV, vaccSTUV, vSTUV);
        }
        vst1q_f32(output, vacc0123);
        vst1q_f32(output + 4, vacc4567);
        vst1q_f32(output + 8, vacc89AB);
        vst1q_f32(output + 12, vaccCDEF);
        vst1q_f32(output + 16, vaccGHIJ);
        vst1q_f32(output + 20, vaccKLMN);
        vst1q_f32(output + 24, vaccOPQR);
        vst1q_f32(output + 28, vaccSTUV);
        output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
      } while (--n != 0);
      output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
      B += 32;
      mc -= 32 * sizeof(float);
    }

  if
    SPARSE_UNLIKELY(mc != 0) {
      output_decrement += 16 * sizeof(float);
      if (mc & (16 * sizeof(float))) {
        const float* w = A;
        const float* b = bias;
        float valpha = alpha;
        float32x4_t vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
        b += 1;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        size_t n = nc;
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vb;
          float32x4_t vacc4567 = vacc0123;
          float32x4_t vacc89AB = vacc0123;
          float32x4_t vaccCDEF = vacc0123;
          vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
          b += 1;
          if
            SPARSE_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                const float32x4_t vi0123 = vld1q_f32(B);
                const float32x4_t vi4567 = vld1q_f32(B + 4);
                const float32x4_t vi89AB = vld1q_f32(B + 8);
                const float32x4_t viCDEF = vld1q_f32(B + 12);
                B = (const float*)((uintptr_t)B + (uintptr_t)diff);
                __builtin_prefetch(B + 16);
                __builtin_prefetch(B + 32);
                const float32x4_t vw = vld1q_dup_f32(w);
                w += 1;
                __builtin_prefetch(w + 32);
                vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
                vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
                vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vw);
                vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vw);
              } while (--nnz != 0);
            }

          if (flag_act == 1) {  // relu
            float32x4_t vzero = vdupq_n_f32(0);
            vacc0123 = vmaxq_f32(vacc0123, vzero);
            vacc4567 = vmaxq_f32(vacc4567, vzero);
            vacc89AB = vmaxq_f32(vacc89AB, vzero);
            vaccCDEF = vmaxq_f32(vaccCDEF, vzero);
          } else if (flag_act == 2) {  // relu6
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            vacc0123 = vmaxq_f32(vacc0123, vzero);
            vacc4567 = vmaxq_f32(vacc4567, vzero);
            vacc89AB = vmaxq_f32(vacc89AB, vzero);
            vaccCDEF = vmaxq_f32(vaccCDEF, vzero);
            vacc0123 = vminq_f32(vacc0123, aph);
            vacc4567 = vminq_f32(vacc4567, aph);
            vacc89AB = vminq_f32(vacc89AB, aph);
            vaccCDEF = vminq_f32(vaccCDEF, aph);
          } else if (flag_act != 0) {  // leaky_relu
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vacc0123, vzero);
            uint32x4_t vflag4567 = vcgeq_f32(vacc4567, vzero);
            uint32x4_t vflag89AB = vcgeq_f32(vacc89AB, vzero);
            uint32x4_t vflagCDEF = vcgeq_f32(vaccCDEF, vzero);
            float32x4_t v0123 = vmulq_f32(vacc0123, aph);
            float32x4_t v4567 = vmulq_f32(vacc4567, aph);
            float32x4_t v89AB = vmulq_f32(vacc89AB, aph);
            float32x4_t vCDEF = vmulq_f32(vaccCDEF, aph);
            vacc0123 = vbslq_f32(vflag0123, vacc0123, v0123);
            vacc4567 = vbslq_f32(vflag4567, vacc4567, v4567);
            vacc89AB = vbslq_f32(vflag89AB, vacc89AB, v89AB);
            vaccCDEF = vbslq_f32(vflagCDEF, vaccCDEF, vCDEF);
          }
          vst1q_f32(output, vacc0123);
          vst1q_f32(output + 4, vacc4567);
          vst1q_f32(output + 8, vacc89AB);
          vst1q_f32(output + 12, vaccCDEF);
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        } while (--n != 0);
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 16;
      }
      output_decrement += 8 * sizeof(float);
      if (mc & (8 * sizeof(float))) {
        const float* w = A;
        const float* b = bias;
        float valpha = alpha;
        float32x4_t vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
        b += 1;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        size_t n = nc;
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vb;
          float32x4_t vacc4567 = vacc0123;
          vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
          b += 1;
          if
            SPARSE_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                const float32x4_t vi0123 = vld1q_f32(B);
                const float32x4_t vi4567 = vld1q_f32(B + 4);
                B = (const float*)((uintptr_t)B + (uintptr_t)diff);
                __builtin_prefetch(B + 16);
                __builtin_prefetch(B + 32);
                const float32x4_t vw = vld1q_dup_f32(w);
                w += 1;
                __builtin_prefetch(w + 32);
                vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
                vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
              } while (--nnz != 0);
            }
          if (flag_act == 1) {  // relu
            float32x4_t vzero = vdupq_n_f32(0);
            vacc0123 = vmaxq_f32(vacc0123, vzero);
            vacc4567 = vmaxq_f32(vacc4567, vzero);
          } else if (flag_act == 2) {  // relu6
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            vacc0123 = vmaxq_f32(vacc0123, vzero);
            vacc4567 = vmaxq_f32(vacc4567, vzero);
            vacc0123 = vminq_f32(vacc0123, aph);
            vacc4567 = vminq_f32(vacc4567, aph);
          } else if (flag_act != 0) {  // leaky_relu
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vacc0123, vzero);
            uint32x4_t vflag4567 = vcgeq_f32(vacc4567, vzero);
            float32x4_t v0123 = vmulq_f32(vacc0123, aph);
            float32x4_t v4567 = vmulq_f32(vacc4567, aph);
            vacc0123 = vbslq_f32(vflag0123, vacc0123, v0123);
            vacc4567 = vbslq_f32(vflag4567, vacc4567, v4567);
          }
          vst1q_f32(output, vacc0123);
          vst1q_f32(output + 4, vacc4567);
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        } while (--n != 0);
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 8;
      }
      output_decrement += 4 * sizeof(float);
      if (mc & (4 * sizeof(float))) {
        const float* w = A;
        const float* b = bias;
        float valpha = alpha;
        float32x4_t vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
        b += 1;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        size_t n = nc;
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vb;
          vb = (has_bias == true) ? vdupq_n_f32(*b) : vdupq_n_f32(0);
          b += 1;
          if
            SPARSE_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                const float32x4_t vi0123 = vld1q_f32(B);
                B = (const float*)((uintptr_t)B + (uintptr_t)diff);
                __builtin_prefetch(B + 16);
                __builtin_prefetch(B + 32);
                const float32x4_t vw = vld1q_dup_f32(w);
                w += 1;
                __builtin_prefetch(w + 32);
                vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
              } while (--nnz != 0);
            }
          if (flag_act == 1) {  // relu
            float32x4_t vzero = vdupq_n_f32(0);
            vacc0123 = vmaxq_f32(vacc0123, vzero);
          } else if (flag_act == 2) {  // relu6
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            vacc0123 = vmaxq_f32(vacc0123, vzero);
            vacc0123 = vminq_f32(vacc0123, aph);
          } else if (flag_act != 0) {  // leaky_relu
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vacc0123, vzero);
            float32x4_t v0123 = vmulq_f32(vacc0123, aph);
            vacc0123 = vbslq_f32(vflag0123, vacc0123, v0123);
          }
          vst1q_f32(output, vacc0123);
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        } while (--n != 0);
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 4;
      }
      output_decrement += 2 * sizeof(float);
      if (mc & (2 * sizeof(float))) {
        const float* w = A;
        const float* b = bias;
        float valpha = alpha;
        float32x2_t vb = (has_bias == true) ? vdup_n_f32(*b) : vdup_n_f32(0);
        b += 1;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        size_t n = nc;
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc01 = vb;
          vb = (has_bias == true) ? vdup_n_f32(*b) : vdup_n_f32(0);
          b += 1;
          if
            SPARSE_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                const float32x2_t vi01 = vld1_f32(B);
                B = (const float*)((uintptr_t)B + (uintptr_t)diff);
                __builtin_prefetch(B + 16);
                __builtin_prefetch(B + 32);
                const float32x2_t vw = vld1_dup_f32(w);
                w += 1;
                __builtin_prefetch(w + 32);
                vacc01 = vmla_f32(vacc01, vi01, vw);
              } while (--nnz != 0);
            }
          if (flag_act == 1) {  // relu
            float32x2_t vzero = vdup_n_f32(0);
            vacc01 = vmax_f32(vacc01, vzero);
          } else if (flag_act == 2) {  // relu6
            float32x2_t vzero = vdup_n_f32(0);
            float32x2_t aph = vdup_n_f32(valpha);
            vacc01 = vmax_f32(vacc01, vzero);
            vacc01 = vmin_f32(vacc01, aph);
          } else if (flag_act != 0) {  // leaky_relu
            float32x2_t vzero = vdup_n_f32(0);
            float32x2_t aph = vdup_n_f32(valpha);
            uint32x2_t vflag0123 = vcge_f32(vacc01, vzero);
            float32x2_t v0123 = vmul_f32(vacc01, aph);
            vacc01 = vbsl_f32(vflag0123, vacc01, v0123);
          }
          vst1_f32(output, vacc01);
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        } while (--n != 0);
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 2;
      }
      output_decrement += 1 * sizeof(float);
      if (mc & (1 * sizeof(float))) {
        const float* w = A;
        const float* b = bias;
        float valpha = alpha;
        float32x2_t vb = (has_bias == true) ? vdup_n_f32(*b) : vdup_n_f32(0);
        b += 1;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        size_t n = nc;
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc0 = vb;
          vb = (has_bias == true) ? vdup_n_f32(*b) : vdup_n_f32(0);
          b += 1;
          if
            SPARSE_LIKELY(nnz != 0) {
              do {
                const intptr_t diff = *dmap++;
                const float32x2_t vi0 = vld1_dup_f32(B);
                B = (const float*)((uintptr_t)B + (uintptr_t)diff);
                __builtin_prefetch(B + 16);
                __builtin_prefetch(B + 32);
                const float32x2_t vw = vld1_dup_f32(w);
                w += 1;
                __builtin_prefetch(w + 32);
                vacc0 = vmla_f32(vacc0, vi0, vw);
              } while (--nnz != 0);
            }
          if (flag_act == 1) {  // relu
            float32x2_t vzero = vdup_n_f32(0);
            vacc0 = vmax_f32(vacc0, vzero);
          } else if (flag_act == 2) {  // relu6
            float32x2_t vzero = vdup_n_f32(0);
            float32x2_t aph = vdup_n_f32(valpha);
            vacc0 = vmax_f32(vacc0, vzero);
            vacc0 = vmin_f32(vacc0, aph);
          } else if (flag_act != 0) {  // leaky_relu
            float32x2_t vzero = vdup_n_f32(0);
            float32x2_t aph = vdup_n_f32(valpha);
            uint32x2_t vflag0123 = vcge_f32(vacc0, vzero);
            float32x2_t v0123 = vmul_f32(vacc0, aph);
            vacc0 = vbsl_f32(vflag0123, vacc0, v0123);
          }
          vst1_lane_f32(output, vacc0, 0);
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        } while (--n != 0);
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 1;
      }
    }
}

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
  bool has_bias = bias != nullptr;
  size_t mc = N * sizeof(int8_t);
  size_t nc = M;
  size_t output_stride = N * sizeof(float);
  size_t output_decrement = output_stride * nc - 32 * sizeof(float);

  while
    SPARSE_LIKELY(mc >= 32 * sizeof(int8_t)) {
      const int8_t* w = A;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      const float* bs = bias;
      const float* sc = scale;
      float valpha = alpha;
      for (size_t i = 0; i < nc; i++) {
        uint32_t nnz = *nnzmap++;
        float vsclae = *sc++;
        float32x4_t vbias =
            (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
        int32x4_t vacc0123 = vdupq_n_s32(0);
        int32x4_t vacc4567 = vacc0123;
        int32x4_t vacc89AB = vacc0123;
        int32x4_t vaccCDEF = vacc0123;
        int32x4_t vaccGHIJ = vacc0123;
        int32x4_t vaccKLMN = vacc0123;
        int32x4_t vaccOPQR = vacc0123;
        int32x4_t vaccSTUV = vacc0123;

        for (size_t j = 0; j < nnz; j++) {
          int8x8_t vi0123 = vld1_s8(B);
          int8x8_t vi4567 = vld1_s8(B + 8);
          int8x8_t vi89AB = vld1_s8(B + 16);
          int8x8_t viCDEF = vld1_s8(B + 24);

          int8x8_t vw = vld1_dup_s8(w);
          w += 1;
          __builtin_prefetch(w + 32);

          intptr_t diff = *dmap++;
          B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
          __builtin_prefetch(B + 16);
          __builtin_prefetch(B + 32);

          int16x8_t vo0123 = vmull_s8(vi0123, vw);
          int16x8_t vo4567 = vmull_s8(vi4567, vw);
          int16x8_t vo89AB = vmull_s8(vi89AB, vw);
          int16x8_t voCDEF = vmull_s8(viCDEF, vw);

          vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
          vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vo0123));
          vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vo4567));
          vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vo4567));
          vaccGHIJ = vaddw_s16(vaccGHIJ, vget_low_s16(vo89AB));
          vaccKLMN = vaddw_s16(vaccKLMN, vget_high_s16(vo89AB));
          vaccOPQR = vaddw_s16(vaccOPQR, vget_low_s16(voCDEF));
          vaccSTUV = vaddw_s16(vaccSTUV, vget_high_s16(voCDEF));
        }

        float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vaccf4567 = vcvtq_f32_s32(vacc4567);
        float32x4_t vaccf89AB = vcvtq_f32_s32(vacc89AB);
        float32x4_t vaccfCDEF = vcvtq_f32_s32(vaccCDEF);
        float32x4_t vaccfGHIJ = vcvtq_f32_s32(vaccGHIJ);
        float32x4_t vaccfKLMN = vcvtq_f32_s32(vaccKLMN);
        float32x4_t vaccfOPQR = vcvtq_f32_s32(vaccOPQR);
        float32x4_t vaccfSTUV = vcvtq_f32_s32(vaccSTUV);

        vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
        vaccf4567 = vmlaq_n_f32(vbias, vaccf4567, vsclae);
        vaccf89AB = vmlaq_n_f32(vbias, vaccf89AB, vsclae);
        vaccfCDEF = vmlaq_n_f32(vbias, vaccfCDEF, vsclae);
        vaccfGHIJ = vmlaq_n_f32(vbias, vaccfGHIJ, vsclae);
        vaccfKLMN = vmlaq_n_f32(vbias, vaccfKLMN, vsclae);
        vaccfOPQR = vmlaq_n_f32(vbias, vaccfOPQR, vsclae);
        vaccfSTUV = vmlaq_n_f32(vbias, vaccfSTUV, vsclae);

        if (flag_act == 1) {  // relu
          float32x4_t vzero = vdupq_n_f32(0);
          vaccf0123 = vmaxq_f32(vaccf0123, vzero);
          vaccf4567 = vmaxq_f32(vaccf4567, vzero);
          vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
          vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
          vaccfGHIJ = vmaxq_f32(vaccfGHIJ, vzero);
          vaccfKLMN = vmaxq_f32(vaccfKLMN, vzero);
          vaccfOPQR = vmaxq_f32(vaccfOPQR, vzero);
          vaccfSTUV = vmaxq_f32(vaccfSTUV, vzero);
        } else if (flag_act == 2) {  // relu6
          float32x4_t vzero = vdupq_n_f32(0);
          float32x4_t aph = vdupq_n_f32(valpha);
          vaccf0123 = vmaxq_f32(vaccf0123, vzero);
          vaccf4567 = vmaxq_f32(vaccf4567, vzero);
          vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
          vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
          vaccfGHIJ = vmaxq_f32(vaccfGHIJ, vzero);
          vaccfKLMN = vmaxq_f32(vaccfKLMN, vzero);
          vaccfOPQR = vmaxq_f32(vaccfOPQR, vzero);
          vaccfSTUV = vmaxq_f32(vaccfSTUV, vzero);
          vaccf0123 = vminq_f32(vaccf0123, aph);
          vaccf4567 = vminq_f32(vaccf4567, aph);
          vaccf89AB = vminq_f32(vaccf89AB, aph);
          vaccfCDEF = vminq_f32(vaccfCDEF, aph);
          vaccfGHIJ = vminq_f32(vaccfGHIJ, aph);
          vaccfKLMN = vminq_f32(vaccfKLMN, aph);
          vaccfOPQR = vminq_f32(vaccfOPQR, aph);
          vaccfSTUV = vminq_f32(vaccfSTUV, aph);
        } else if (flag_act != 0) {  // leaky_relu
          float32x4_t vzero = vdupq_n_f32(0);
          float32x4_t aph = vdupq_n_f32(valpha);
          uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
          uint32x4_t vflag4567 = vcgeq_f32(vaccf4567, vzero);
          uint32x4_t vflag89AB = vcgeq_f32(vaccf89AB, vzero);
          uint32x4_t vflagCDEF = vcgeq_f32(vaccfCDEF, vzero);
          uint32x4_t vflagGHIJ = vcgeq_f32(vaccfGHIJ, vzero);
          uint32x4_t vflagKLMN = vcgeq_f32(vaccfKLMN, vzero);
          uint32x4_t vflagOPQR = vcgeq_f32(vaccfOPQR, vzero);
          uint32x4_t vflagSTUV = vcgeq_f32(vaccfSTUV, vzero);
          float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
          float32x4_t v4567 = vmulq_f32(vaccf4567, aph);
          float32x4_t v89AB = vmulq_f32(vaccf89AB, aph);
          float32x4_t vCDEF = vmulq_f32(vaccfCDEF, aph);
          float32x4_t vGHIJ = vmulq_f32(vaccfGHIJ, aph);
          float32x4_t vKLMN = vmulq_f32(vaccfKLMN, aph);
          float32x4_t vOPQR = vmulq_f32(vaccfOPQR, aph);
          float32x4_t vSTUV = vmulq_f32(vaccfSTUV, aph);
          vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
          vaccf4567 = vbslq_f32(vflag4567, vaccf4567, v4567);
          vaccf89AB = vbslq_f32(vflag89AB, vaccf89AB, v89AB);
          vaccfCDEF = vbslq_f32(vflagCDEF, vaccfCDEF, vCDEF);
          vaccfGHIJ = vbslq_f32(vflagGHIJ, vaccfGHIJ, vGHIJ);
          vaccfKLMN = vbslq_f32(vflagKLMN, vaccfKLMN, vKLMN);
          vaccfOPQR = vbslq_f32(vflagOPQR, vaccfOPQR, vOPQR);
          vaccfSTUV = vbslq_f32(vflagSTUV, vaccfSTUV, vSTUV);
        }
        vst1q_f32(output, vaccf0123);
        vst1q_f32(output + 4, vaccf4567);
        vst1q_f32(output + 8, vaccf89AB);
        vst1q_f32(output + 12, vaccfCDEF);
        vst1q_f32(output + 16, vaccfGHIJ);
        vst1q_f32(output + 20, vaccfKLMN);
        vst1q_f32(output + 24, vaccfOPQR);
        vst1q_f32(output + 28, vaccfSTUV);

        output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
      }
      output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
      B += 32;
      mc -= 32 * sizeof(int8_t);
    }
  if
    SPARSE_UNLIKELY(mc != 0) {
      output_decrement += 16 * sizeof(float);
      if (mc & (16 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* bs = bias;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float32x4_t vbias =
              (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
          int32x4_t vacc0123 = vdupq_n_s32(0);
          int32x4_t vacc4567 = vacc0123;
          int32x4_t vacc89AB = vacc0123;
          int32x4_t vaccCDEF = vacc0123;

          for (size_t j = 0; j < nnz; j++) {
            int8x8_t vi0123 = vld1_s8(B);
            int8x8_t vi4567 = vld1_s8(B + 8);

            int8x8_t vw = vld1_dup_s8(w);
            w += 1;
            __builtin_prefetch(w + 32);
            intptr_t diff = *dmap++;
            B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
            __builtin_prefetch(B + 16);
            __builtin_prefetch(B + 32);

            int16x8_t vo0123 = vmull_s8(vi0123, vw);
            int16x8_t vo4567 = vmull_s8(vi4567, vw);

            vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
            vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vo0123));
            vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vo4567));
            vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vo4567));
          }

          float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
          float32x4_t vaccf4567 = vcvtq_f32_s32(vacc4567);
          float32x4_t vaccf89AB = vcvtq_f32_s32(vacc89AB);
          float32x4_t vaccfCDEF = vcvtq_f32_s32(vaccCDEF);

          vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
          vaccf4567 = vmlaq_n_f32(vbias, vaccf4567, vsclae);
          vaccf89AB = vmlaq_n_f32(vbias, vaccf89AB, vsclae);
          vaccfCDEF = vmlaq_n_f32(vbias, vaccfCDEF, vsclae);

          if (flag_act == 1) {
            float32x4_t vzero = vdupq_n_f32(0);
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
            vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
            vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
          } else if (flag_act == 2) {
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
            vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
            vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
            vaccf0123 = vminq_f32(vaccf0123, aph);
            vaccf4567 = vminq_f32(vaccf4567, aph);
            vaccf89AB = vminq_f32(vaccf89AB, aph);
            vaccfCDEF = vminq_f32(vaccfCDEF, aph);
          } else if (flag_act != 0) {
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
            uint32x4_t vflag4567 = vcgeq_f32(vaccf4567, vzero);
            uint32x4_t vflag89AB = vcgeq_f32(vaccf89AB, vzero);
            uint32x4_t vflagCDEF = vcgeq_f32(vaccfCDEF, vzero);
            float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
            float32x4_t v4567 = vmulq_f32(vaccf4567, aph);
            float32x4_t v89AB = vmulq_f32(vaccf89AB, aph);
            float32x4_t vCDEF = vmulq_f32(vaccfCDEF, aph);
            vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
            vaccf4567 = vbslq_f32(vflag4567, vaccf4567, v4567);
            vaccf89AB = vbslq_f32(vflag89AB, vaccf89AB, v89AB);
            vaccfCDEF = vbslq_f32(vflagCDEF, vaccfCDEF, vCDEF);
          }

          vst1q_f32(output, vaccf0123);
          vst1q_f32(output + 4, vaccf4567);
          vst1q_f32(output + 8, vaccf89AB);
          vst1q_f32(output + 12, vaccfCDEF);

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
        const float* bs = bias;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float32x4_t vbias =
              (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
          int32x4_t vacc0123 = vdupq_n_s32(0);
          int32x4_t vacc4567 = vacc0123;

          for (size_t j = 0; j < nnz; j++) {
            int8x8_t vi0123 = vld1_s8(B);
            int8x8_t vw = vld1_dup_s8(w);
            w += 1;
            __builtin_prefetch(w + 32);
            intptr_t diff = *dmap++;
            B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
            __builtin_prefetch(B + 16);
            __builtin_prefetch(B + 32);
            int16x8_t vo0123 = vmull_s8(vi0123, vw);
            vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
            vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vo0123));
          }
          float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
          float32x4_t vaccf4567 = vcvtq_f32_s32(vacc4567);
          vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
          vaccf4567 = vmlaq_n_f32(vbias, vaccf4567, vsclae);
          if (flag_act == 1) {
            float32x4_t vzero = vdupq_n_f32(0);
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
          } else if (flag_act == 2) {
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
            vaccf0123 = vminq_f32(vaccf0123, aph);
            vaccf4567 = vminq_f32(vaccf4567, aph);
          } else if (flag_act != 0) {
            float32x4_t vzero = vdupq_n_f32(0);
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
            uint32x4_t vflag4567 = vcgeq_f32(vaccf4567, vzero);
            float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
            float32x4_t v4567 = vmulq_f32(vaccf4567, aph);
            vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
            vaccf4567 = vbslq_f32(vflag4567, vaccf4567, v4567);
          }
          vst1q_f32(output, vaccf0123);
          vst1q_f32(output + 4, vaccf4567);
          output = reinterpret_cast<float*>((uintptr_t)output + output_stride);
        }
        output = reinterpret_cast<float*>((uintptr_t)output - output_decrement);
        B += 8;
        mc -= 8 * sizeof(int8_t);
      }
      output_decrement += 4 * sizeof(float);
      if
        SPARSE_UNLIKELY(mc >= 4 && mc < 8 * sizeof(int8_t)) {
          const int8_t* w = A;
          const int32_t* dmap = widx_dmap;
          const uint32_t* nnzmap = nidx_nnzmap;
          const float* bs = bias;
          const float* sc = scale;
          float valpha = alpha;
          for (int i = 0; i < nc; i++) {
            uint32_t nnz = *nnzmap++;
            float vsclae = *sc++;
            float32x4_t vbias =
                (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
            int32x4_t vacc0123 = vdupq_n_s32(0);
            for (int j = 0; j < nnz; j++) {
              int8x8_t vi0123 = vdup_n_s8(0);
              vi0123 = vld1_lane_s8(B, vi0123, 0);
              vi0123 = vld1_lane_s8(B + 1, vi0123, 1);
              vi0123 = vld1_lane_s8(B + 2, vi0123, 2);
              vi0123 = vld1_lane_s8(B + 3, vi0123, 3);
              int8x8_t vw = vld1_dup_s8(w);
              w += 1;
              __builtin_prefetch(w + 32);
              intptr_t diff = *dmap++;
              B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
              __builtin_prefetch(B + 16);
              __builtin_prefetch(B + 32);
              int16x8_t vo0123 = vmull_s8(vi0123, vw);
              vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
            }
            float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
            vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
            if (flag_act == 1) {
              float32x4_t vzero = vdupq_n_f32(0);
              vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            } else if (flag_act == 2) {
              float32x4_t vzero = vdupq_n_f32(0);
              float32x4_t aph = vdupq_n_f32(valpha);
              vaccf0123 = vmaxq_f32(vaccf0123, vzero);
              vaccf0123 = vminq_f32(vaccf0123, aph);
            } else if (flag_act != 0) {
              float32x4_t vzero = vdupq_n_f32(0);
              float32x4_t aph = vdupq_n_f32(valpha);
              uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
              float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
              vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
            }
            vst1q_f32(output, vaccf0123);
            output =
                reinterpret_cast<float*>((uintptr_t)output + output_stride);
          }
          output =
              reinterpret_cast<float*>((uintptr_t)output - output_decrement);
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
          float val = alpha;
          for (size_t i = 0; i < nc; i++) {
            float vbias = (has_bias == true) ? *bs++ : 0;
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
            for (size_t k = 0; k < mc; k++) {
              *(output + k) = *(output + k) * vscale + vbias;
              switch (flag_act) {
                case 0:
                  break;
                case 1:  // relu
                  *(output + k) = *(output + k) > 0 ? *(output + k) : 0;
                  break;
                case 2:  // relu6
                  *(output + k) = *(output + k) > 0 ? *(output + k) : 0;
                  *(output + k) = *(output + k) < val ? *(output + k) : val;
                  break;
                default:  // leaky_relu
                  *(output + k) =
                      *(output + k) >= 0 ? *(output + k) : *(output + k) * val;
                  break;
              }
            }
            output =
                reinterpret_cast<float*>((uintptr_t)output + output_stride);
          }
        }
    }
}

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
  bool has_bias = bias != nullptr;
  size_t mc = N * sizeof(int8_t);
  size_t nc = M;
  size_t output_stride = N * sizeof(int8_t);
  size_t output_decrement = output_stride * nc - 32 * sizeof(int8_t);
  while
    SPARSE_LIKELY(mc >= 32 * sizeof(int8_t)) {
      const int8_t* w = A;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      const float* bs = bias;
      const float* sc = scale;
      float valpha = alpha;

      for (size_t i = 0; i < nc; i++) {
        uint32_t nnz = *nnzmap++;
        float vsclae = *sc++;
        float32x4_t vbias =
            (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
        int32x4_t vacc0123 = vdupq_n_s32(0);
        int32x4_t vacc4567 = vacc0123;
        int32x4_t vacc89AB = vacc0123;
        int32x4_t vaccCDEF = vacc0123;
        int32x4_t vaccGHIJ = vacc0123;
        int32x4_t vaccKLMN = vacc0123;
        int32x4_t vaccOPQR = vacc0123;
        int32x4_t vaccSTUV = vacc0123;

        for (size_t j = 0; j < nnz; j++) {
          int8x8_t vi0123 = vld1_s8(B);
          int8x8_t vi4567 = vld1_s8(B + 8);
          int8x8_t vi89AB = vld1_s8(B + 16);
          int8x8_t viCDEF = vld1_s8(B + 24);

          int8x8_t vw = vld1_dup_s8(w);
          w += 1;
          __builtin_prefetch(w + 32);

          intptr_t diff = *dmap++;
          B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
          __builtin_prefetch(B + 16);
          __builtin_prefetch(B + 32);

          int16x8_t vo0123 = vmull_s8(vi0123, vw);
          int16x8_t vo4567 = vmull_s8(vi4567, vw);
          int16x8_t vo89AB = vmull_s8(vi89AB, vw);
          int16x8_t voCDEF = vmull_s8(viCDEF, vw);

          vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
          vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vo0123));
          vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vo4567));
          vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vo4567));
          vaccGHIJ = vaddw_s16(vaccGHIJ, vget_low_s16(vo89AB));
          vaccKLMN = vaddw_s16(vaccKLMN, vget_high_s16(vo89AB));
          vaccOPQR = vaddw_s16(vaccOPQR, vget_low_s16(voCDEF));
          vaccSTUV = vaddw_s16(vaccSTUV, vget_high_s16(voCDEF));
        }

        float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vaccf4567 = vcvtq_f32_s32(vacc4567);
        float32x4_t vaccf89AB = vcvtq_f32_s32(vacc89AB);
        float32x4_t vaccfCDEF = vcvtq_f32_s32(vaccCDEF);
        float32x4_t vaccfGHIJ = vcvtq_f32_s32(vaccGHIJ);
        float32x4_t vaccfKLMN = vcvtq_f32_s32(vaccKLMN);
        float32x4_t vaccfOPQR = vcvtq_f32_s32(vaccOPQR);
        float32x4_t vaccfSTUV = vcvtq_f32_s32(vaccSTUV);

        vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
        vaccf4567 = vmlaq_n_f32(vbias, vaccf4567, vsclae);
        vaccf89AB = vmlaq_n_f32(vbias, vaccf89AB, vsclae);
        vaccfCDEF = vmlaq_n_f32(vbias, vaccfCDEF, vsclae);
        vaccfGHIJ = vmlaq_n_f32(vbias, vaccfGHIJ, vsclae);
        vaccfKLMN = vmlaq_n_f32(vbias, vaccfKLMN, vsclae);
        vaccfOPQR = vmlaq_n_f32(vbias, vaccfOPQR, vsclae);
        vaccfSTUV = vmlaq_n_f32(vbias, vaccfSTUV, vsclae);

        float32x4_t vzero = vdupq_n_f32(0);
        if (flag_act == 1) {  // relu
          vaccf0123 = vmaxq_f32(vaccf0123, vzero);
          vaccf4567 = vmaxq_f32(vaccf4567, vzero);
          vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
          vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
          vaccfGHIJ = vmaxq_f32(vaccfGHIJ, vzero);
          vaccfKLMN = vmaxq_f32(vaccfKLMN, vzero);
          vaccfOPQR = vmaxq_f32(vaccfOPQR, vzero);
          vaccfSTUV = vmaxq_f32(vaccfSTUV, vzero);
        } else if (flag_act == 2) {  // relu6
          float32x4_t aph = vdupq_n_f32(valpha);
          vaccf0123 = vmaxq_f32(vaccf0123, vzero);
          vaccf4567 = vmaxq_f32(vaccf4567, vzero);
          vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
          vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
          vaccfGHIJ = vmaxq_f32(vaccfGHIJ, vzero);
          vaccfKLMN = vmaxq_f32(vaccfKLMN, vzero);
          vaccfOPQR = vmaxq_f32(vaccfOPQR, vzero);
          vaccfSTUV = vmaxq_f32(vaccfSTUV, vzero);
          vaccf0123 = vminq_f32(vaccf0123, aph);
          vaccf4567 = vminq_f32(vaccf4567, aph);
          vaccf89AB = vminq_f32(vaccf89AB, aph);
          vaccfCDEF = vminq_f32(vaccfCDEF, aph);
          vaccfGHIJ = vminq_f32(vaccfGHIJ, aph);
          vaccfKLMN = vminq_f32(vaccfKLMN, aph);
          vaccfOPQR = vminq_f32(vaccfOPQR, aph);
          vaccfSTUV = vminq_f32(vaccfSTUV, aph);
        } else if (flag_act != 0) {  // leaky_relu
          float32x4_t aph = vdupq_n_f32(valpha);
          uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
          uint32x4_t vflag4567 = vcgeq_f32(vaccf4567, vzero);
          uint32x4_t vflag89AB = vcgeq_f32(vaccf89AB, vzero);
          uint32x4_t vflagCDEF = vcgeq_f32(vaccfCDEF, vzero);
          uint32x4_t vflagGHIJ = vcgeq_f32(vaccfGHIJ, vzero);
          uint32x4_t vflagKLMN = vcgeq_f32(vaccfKLMN, vzero);
          uint32x4_t vflagOPQR = vcgeq_f32(vaccfOPQR, vzero);
          uint32x4_t vflagSTUV = vcgeq_f32(vaccfSTUV, vzero);
          float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
          float32x4_t v4567 = vmulq_f32(vaccf4567, aph);
          float32x4_t v89AB = vmulq_f32(vaccf89AB, aph);
          float32x4_t vCDEF = vmulq_f32(vaccfCDEF, aph);
          float32x4_t vGHIJ = vmulq_f32(vaccfGHIJ, aph);
          float32x4_t vKLMN = vmulq_f32(vaccfKLMN, aph);
          float32x4_t vOPQR = vmulq_f32(vaccfOPQR, aph);
          float32x4_t vSTUV = vmulq_f32(vaccfSTUV, aph);
          vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
          vaccf4567 = vbslq_f32(vflag4567, vaccf4567, v4567);
          vaccf89AB = vbslq_f32(vflag89AB, vaccf89AB, v89AB);
          vaccfCDEF = vbslq_f32(vflagCDEF, vaccfCDEF, vCDEF);
          vaccfGHIJ = vbslq_f32(vflagGHIJ, vaccfGHIJ, vGHIJ);
          vaccfKLMN = vbslq_f32(vflagKLMN, vaccfKLMN, vKLMN);
          vaccfOPQR = vbslq_f32(vflagOPQR, vaccfOPQR, vOPQR);
          vaccfSTUV = vbslq_f32(vflagSTUV, vaccfSTUV, vSTUV);
        }

        float32x4_t vpos = vdupq_n_f32(0.5);
        float32x4_t vneg = vdupq_n_f32(-0.5);
        vaccf0123 = vbslq_f32(vcgeq_f32(vaccf0123, vzero),
                              vaddq_f32(vaccf0123, vpos),
                              vaddq_f32(vaccf0123, vneg));
        vaccf4567 = vbslq_f32(vcgeq_f32(vaccf4567, vzero),
                              vaddq_f32(vaccf4567, vpos),
                              vaddq_f32(vaccf4567, vneg));
        vaccf89AB = vbslq_f32(vcgeq_f32(vaccf89AB, vzero),
                              vaddq_f32(vaccf89AB, vpos),
                              vaddq_f32(vaccf89AB, vneg));
        vaccfCDEF = vbslq_f32(vcgeq_f32(vaccfCDEF, vzero),
                              vaddq_f32(vaccfCDEF, vpos),
                              vaddq_f32(vaccfCDEF, vneg));
        vaccfGHIJ = vbslq_f32(vcgeq_f32(vaccfGHIJ, vzero),
                              vaddq_f32(vaccfGHIJ, vpos),
                              vaddq_f32(vaccfGHIJ, vneg));
        vaccfKLMN = vbslq_f32(vcgeq_f32(vaccfKLMN, vzero),
                              vaddq_f32(vaccfKLMN, vpos),
                              vaddq_f32(vaccfKLMN, vneg));
        vaccfOPQR = vbslq_f32(vcgeq_f32(vaccfOPQR, vzero),
                              vaddq_f32(vaccfOPQR, vpos),
                              vaddq_f32(vaccfOPQR, vneg));
        vaccfSTUV = vbslq_f32(vcgeq_f32(vaccfSTUV, vzero),
                              vaddq_f32(vaccfSTUV, vpos),
                              vaddq_f32(vaccfSTUV, vneg));

        int32x4_t vacci0123 = vcvtq_s32_f32(vaccf0123);
        int32x4_t vacci4567 = vcvtq_s32_f32(vaccf4567);
        int32x4_t vacci89AB = vcvtq_s32_f32(vaccf89AB);
        int32x4_t vacciCDEF = vcvtq_s32_f32(vaccfCDEF);
        int32x4_t vacciGHIJ = vcvtq_s32_f32(vaccfGHIJ);
        int32x4_t vacciKLMN = vcvtq_s32_f32(vaccfKLMN);
        int32x4_t vacciOPQR = vcvtq_s32_f32(vaccfOPQR);
        int32x4_t vacciSTUV = vcvtq_s32_f32(vaccfSTUV);

        int16x4_t v16i0123 = vqmovn_s32(vacci0123);
        int16x4_t v16i4567 = vqmovn_s32(vacci4567);
        int16x4_t v16i89AB = vqmovn_s32(vacci89AB);
        int16x4_t v16iCDEF = vqmovn_s32(vacciCDEF);
        int16x4_t v16iGHIJ = vqmovn_s32(vacciGHIJ);
        int16x4_t v16iKLMN = vqmovn_s32(vacciKLMN);
        int16x4_t v16iOPQR = vqmovn_s32(vacciOPQR);
        int16x4_t v16iSTUV = vqmovn_s32(vacciSTUV);

        int8x8_t v8i01234567 = vqmovn_s16(vcombine_s16(v16i0123, v16i4567));
        int8x8_t v8i89ABCDEF = vqmovn_s16(vcombine_s16(v16i89AB, v16iCDEF));
        int8x8_t v8iGHIJKLMN = vqmovn_s16(vcombine_s16(v16iGHIJ, v16iKLMN));
        int8x8_t v8iOPQRSTUV = vqmovn_s16(vcombine_s16(v16iOPQR, v16iSTUV));

        vst1_s8(output, v8i01234567);
        vst1_s8(output + 8, v8i89ABCDEF);
        vst1_s8(output + 16, v8iGHIJKLMN);
        vst1_s8(output + 24, v8iOPQRSTUV);

        output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
      }
      output = reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
      B += 32;
      mc -= 32 * sizeof(int8_t);
    }
  if
    SPARSE_UNLIKELY(mc != 0) {
      output_decrement += 16 * sizeof(int8_t);
      if (mc & (16 * sizeof(int8_t))) {
        const int8_t* w = A;
        const int32_t* dmap = widx_dmap;
        const uint32_t* nnzmap = nidx_nnzmap;
        const float* bs = bias;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float32x4_t vbias =
              (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
          int32x4_t vacc0123 = vdupq_n_s32(0);
          int32x4_t vacc4567 = vacc0123;
          int32x4_t vacc89AB = vacc0123;
          int32x4_t vaccCDEF = vacc0123;

          for (size_t j = 0; j < nnz; j++) {
            int8x8_t vi0123 = vld1_s8(B);
            int8x8_t vi4567 = vld1_s8(B + 8);

            int8x8_t vw = vld1_dup_s8(w);
            w += 1;
            __builtin_prefetch(w + 32);
            intptr_t diff = *dmap++;
            B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
            __builtin_prefetch(B + 16);
            __builtin_prefetch(B + 32);

            int16x8_t vo0123 = vmull_s8(vi0123, vw);
            int16x8_t vo4567 = vmull_s8(vi4567, vw);

            vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
            vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vo0123));
            vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vo4567));
            vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vo4567));
          }

          float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
          float32x4_t vaccf4567 = vcvtq_f32_s32(vacc4567);
          float32x4_t vaccf89AB = vcvtq_f32_s32(vacc89AB);
          float32x4_t vaccfCDEF = vcvtq_f32_s32(vaccCDEF);

          vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
          vaccf4567 = vmlaq_n_f32(vbias, vaccf4567, vsclae);
          vaccf89AB = vmlaq_n_f32(vbias, vaccf89AB, vsclae);
          vaccfCDEF = vmlaq_n_f32(vbias, vaccfCDEF, vsclae);

          float32x4_t vzero = vdupq_n_f32(0);
          if (flag_act == 1) {
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
            vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
            vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
          } else if (flag_act == 2) {
            float32x4_t aph = vdupq_n_f32(valpha);
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
            vaccf89AB = vmaxq_f32(vaccf89AB, vzero);
            vaccfCDEF = vmaxq_f32(vaccfCDEF, vzero);
            vaccf0123 = vminq_f32(vaccf0123, aph);
            vaccf4567 = vminq_f32(vaccf4567, aph);
            vaccf89AB = vminq_f32(vaccf89AB, aph);
            vaccfCDEF = vminq_f32(vaccfCDEF, aph);
          } else if (flag_act != 0) {
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
            uint32x4_t vflag4567 = vcgeq_f32(vaccf4567, vzero);
            uint32x4_t vflag89AB = vcgeq_f32(vaccf89AB, vzero);
            uint32x4_t vflagCDEF = vcgeq_f32(vaccfCDEF, vzero);
            float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
            float32x4_t v4567 = vmulq_f32(vaccf4567, aph);
            float32x4_t v89AB = vmulq_f32(vaccf89AB, aph);
            float32x4_t vCDEF = vmulq_f32(vaccfCDEF, aph);
            vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
            vaccf4567 = vbslq_f32(vflag4567, vaccf4567, v4567);
            vaccf89AB = vbslq_f32(vflag89AB, vaccf89AB, v89AB);
            vaccfCDEF = vbslq_f32(vflagCDEF, vaccfCDEF, vCDEF);
          }

          float32x4_t vpos = vdupq_n_f32(0.5);
          float32x4_t vneg = vdupq_n_f32(-0.5);
          vaccf0123 = vbslq_f32(vcgeq_f32(vaccf0123, vzero),
                                vaddq_f32(vaccf0123, vpos),
                                vaddq_f32(vaccf0123, vneg));
          vaccf4567 = vbslq_f32(vcgeq_f32(vaccf4567, vzero),
                                vaddq_f32(vaccf4567, vpos),
                                vaddq_f32(vaccf4567, vneg));
          vaccf89AB = vbslq_f32(vcgeq_f32(vaccf89AB, vzero),
                                vaddq_f32(vaccf89AB, vpos),
                                vaddq_f32(vaccf89AB, vneg));
          vaccfCDEF = vbslq_f32(vcgeq_f32(vaccfCDEF, vzero),
                                vaddq_f32(vaccfCDEF, vpos),
                                vaddq_f32(vaccfCDEF, vneg));

          int32x4_t vacci0123 = vcvtq_s32_f32(vaccf0123);
          int32x4_t vacci4567 = vcvtq_s32_f32(vaccf4567);
          int32x4_t vacci89AB = vcvtq_s32_f32(vaccf89AB);
          int32x4_t vacciCDEF = vcvtq_s32_f32(vaccfCDEF);

          int16x4_t v16i0123 = vqmovn_s32(vacci0123);
          int16x4_t v16i4567 = vqmovn_s32(vacci4567);
          int16x4_t v16i89AB = vqmovn_s32(vacci89AB);
          int16x4_t v16iCDEF = vqmovn_s32(vacciCDEF);

          int8x8_t v8i01234567 = vqmovn_s16(vcombine_s16(v16i0123, v16i4567));
          int8x8_t v8i89ABCDEF = vqmovn_s16(vcombine_s16(v16i89AB, v16iCDEF));

          vst1_s8(output, v8i01234567);
          vst1_s8(output + 8, v8i89ABCDEF);

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
        const float* bs = bias;
        const float* sc = scale;
        float valpha = alpha;

        for (size_t i = 0; i < nc; i++) {
          uint32_t nnz = *nnzmap++;
          float vsclae = *sc++;
          float32x4_t vbias =
              (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
          int32x4_t vacc0123 = vdupq_n_s32(0);
          int32x4_t vacc4567 = vacc0123;

          for (size_t j = 0; j < nnz; j++) {
            int8x8_t vi0123 = vld1_s8(B);
            int8x8_t vw = vld1_dup_s8(w);
            w += 1;
            __builtin_prefetch(w + 32);
            intptr_t diff = *dmap++;
            B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
            __builtin_prefetch(B + 16);
            __builtin_prefetch(B + 32);

            int16x8_t vo0123 = vmull_s8(vi0123, vw);

            vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
            vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vo0123));
          }
          float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
          float32x4_t vaccf4567 = vcvtq_f32_s32(vacc4567);

          vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
          vaccf4567 = vmlaq_n_f32(vbias, vaccf4567, vsclae);

          float32x4_t vzero = vdupq_n_f32(0);
          if (flag_act == 1) {
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
          } else if (flag_act == 2) {
            float32x4_t aph = vdupq_n_f32(valpha);
            vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            vaccf4567 = vmaxq_f32(vaccf4567, vzero);
            vaccf0123 = vminq_f32(vaccf0123, aph);
            vaccf4567 = vminq_f32(vaccf4567, aph);
          } else if (flag_act != 0) {
            float32x4_t aph = vdupq_n_f32(valpha);
            uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
            uint32x4_t vflag4567 = vcgeq_f32(vaccf4567, vzero);
            float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
            float32x4_t v4567 = vmulq_f32(vaccf4567, aph);
            vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
            vaccf4567 = vbslq_f32(vflag4567, vaccf4567, v4567);
          }

          float32x4_t vpos = vdupq_n_f32(0.5);
          float32x4_t vneg = vdupq_n_f32(-0.5);
          vaccf0123 = vbslq_f32(vcgeq_f32(vaccf0123, vzero),
                                vaddq_f32(vaccf0123, vpos),
                                vaddq_f32(vaccf0123, vneg));
          vaccf4567 = vbslq_f32(vcgeq_f32(vaccf4567, vzero),
                                vaddq_f32(vaccf4567, vpos),
                                vaddq_f32(vaccf4567, vneg));

          int32x4_t vacci0123 = vcvtq_s32_f32(vaccf0123);
          int32x4_t vacci4567 = vcvtq_s32_f32(vaccf4567);

          int16x4_t v16i0123 = vqmovn_s32(vacci0123);
          int16x4_t v16i4567 = vqmovn_s32(vacci4567);

          int8x8_t v8i01234567 = vqmovn_s16(vcombine_s16(v16i0123, v16i4567));

          vst1_s8(output, v8i01234567);

          output = reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
        }
        output =
            reinterpret_cast<int8_t*>((uintptr_t)output - output_decrement);
        B += 8;
        mc -= 8 * sizeof(int8_t);
      }
      output_decrement += 4 * sizeof(int8_t);
      if
        SPARSE_UNLIKELY(mc >= 4 && mc < 8 * sizeof(int8_t)) {
          const int8_t* w = A;
          const int32_t* dmap = widx_dmap;
          const uint32_t* nnzmap = nidx_nnzmap;
          const float* bs = bias;
          const float* sc = scale;
          float valpha = alpha;

          for (int i = 0; i < nc; i++) {
            uint32_t nnz = *nnzmap++;
            float vsclae = *sc++;
            float32x4_t vbias =
                (has_bias == true) ? vdupq_n_f32(*bs++) : vdupq_n_f32(0);
            int32x4_t vacc0123 = vdupq_n_s32(0);
            for (int j = 0; j < nnz; j++) {
              int8x8_t vi0123 = vdup_n_s8(0);
              vi0123 = vld1_lane_s8(B, vi0123, 0);
              vi0123 = vld1_lane_s8(B + 1, vi0123, 1);
              vi0123 = vld1_lane_s8(B + 2, vi0123, 2);
              vi0123 = vld1_lane_s8(B + 3, vi0123, 3);

              int8x8_t vw = vld1_dup_s8(w);
              w += 1;
              __builtin_prefetch(w + 32);
              intptr_t diff = *dmap++;
              B = (const int8_t*)((uintptr_t)B + (uintptr_t)diff);
              __builtin_prefetch(B + 16);
              __builtin_prefetch(B + 32);

              int16x8_t vo0123 = vmull_s8(vi0123, vw);
              vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vo0123));
            }
            float32x4_t vaccf0123 = vcvtq_f32_s32(vacc0123);
            vaccf0123 = vmlaq_n_f32(vbias, vaccf0123, vsclae);
            float32x4_t vzero = vdupq_n_f32(0);
            if (flag_act == 1) {
              vaccf0123 = vmaxq_f32(vaccf0123, vzero);
            } else if (flag_act == 2) {
              float32x4_t aph = vdupq_n_f32(valpha);
              vaccf0123 = vmaxq_f32(vaccf0123, vzero);
              vaccf0123 = vminq_f32(vaccf0123, aph);
            } else if (flag_act != 0) {
              float32x4_t aph = vdupq_n_f32(valpha);
              uint32x4_t vflag0123 = vcgeq_f32(vaccf0123, vzero);
              float32x4_t v0123 = vmulq_f32(vaccf0123, aph);
              vaccf0123 = vbslq_f32(vflag0123, vaccf0123, v0123);
            }

            float32x4_t vpos = vdupq_n_f32(0.5);
            float32x4_t vneg = vdupq_n_f32(-0.5);
            vaccf0123 = vbslq_f32(vcgeq_f32(vaccf0123, vzero),
                                  vaddq_f32(vaccf0123, vpos),
                                  vaddq_f32(vaccf0123, vneg));

            int32x4_t vacci0123 = vcvtq_s32_f32(vaccf0123);

            int16x4_t v16i0123 = vqmovn_s32(vacci0123);
            int16x4_t v16i4567 = vdup_n_s16(0);
            int8x8_t v8i01234567 = vqmovn_s16(vcombine_s16(v16i0123, v16i4567));

            vst1_lane_s8(output, v8i01234567, 0);
            vst1_lane_s8(output + 1, v8i01234567, 1);
            vst1_lane_s8(output + 2, v8i01234567, 2);
            vst1_lane_s8(output + 3, v8i01234567, 3);

            output =
                reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
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
          float val = alpha;

          for (size_t i = 0; i < nc; i++) {
            float vbias = (has_bias == true) ? *bs++ : 0;
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
              out[k] = out[k] >= 0 ? out[k] + 0.5 : out[k] - 0.5;
              float vax = out[k] > -127.0 ? out[k] : -127.0;
              vax = out[k] > 127.0 ? 127.0 : out[k];
              *(output + k) = static_cast<int8_t>(vax);
            }
            output =
                reinterpret_cast<int8_t*>((uintptr_t)output + output_stride);
          }
        }
    }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
