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
      // local_alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  bool has_bias = param.bias != nullptr;
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

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
