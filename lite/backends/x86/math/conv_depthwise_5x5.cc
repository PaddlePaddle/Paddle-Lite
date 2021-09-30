/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <immintrin.h>
#include <vector>
#include "lite/backends/x86/math/avx/avx_mathfuns.h"
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/backends/x86/math/conv_depthwise_impl.h"
#include "lite/core/memory.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {
#define Min(a, b) (a < b ? a : b)
#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
void conv_depthwise_5x5s1(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          bool flag_bias,
                          const operators::ActivationParam act_param) {
  bool has_active = act_param.has_active;
  auto act_type = act_param.active_type;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int block_channel = 8;
  int in_len = block_channel * (2 * pad + w_in);

  int channel_num = ROUNDUP(ch_in, block_channel);
  float* pack_weight = static_cast<float*>(TargetMalloc(
      TARGET(kX86),
      1 * channel_num / block_channel * 5 * 5 * block_channel * sizeof(float)));
  float* pack_input = static_cast<float*>(TargetMalloc(
      TARGET(kX86),
      1 * (h_in + 2 * pad) * (w_in + 2 * pad) * block_channel * sizeof(float)));
  float* pack_out = static_cast<float*>(TargetMalloc(
      TARGET(kX86), 1 * h_out * w_out * block_channel * sizeof(float)));

  packC8_with_Cleft(weights, pack_weight, {0, 0, 0, 0}, 5, 5, ch_in);

  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_out * size_out_channel;

    for (int c = 0; c < ch_out; c += block_channel) {
      int real_block_channel = Min(block_channel, ch_out - c);
      auto* dout_ptr = dout_batch + c * size_out_channel;
      auto* din_ptr = din_batch + c * size_in_channel;
      auto* weights_data = pack_weight + c * 5 * 5;
      packC8_with_Cleft(din_ptr,
                        pack_input,
                        {pad, pad, pad, pad},
                        h_in,
                        w_in,
                        real_block_channel);
      int mask_ptr[8] = {0};
      for (int i = 0; i < 8; i++) {
        if (real_block_channel > i) {
          mask_ptr[i] = 0x80000000;
        }
      }
      __m256i bias_mask = _mm256_set_epi32(mask_ptr[7],
                                           mask_ptr[6],
                                           mask_ptr[5],
                                           mask_ptr[4],
                                           mask_ptr[3],
                                           mask_ptr[2],
                                           mask_ptr[1],
                                           mask_ptr[0]);
      __m256 _bias = flag_bias ? _mm256_maskload_ps(bias + c, bias_mask)
                               : _mm256_set1_ps(0.f);

      for (int i = 0; i < h_out; i++) {
        const float* block_inr0 = pack_input + i * in_len;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        int j = 0;
        float* dout_block = pack_out + i * w_out * 8;
        for (; j + 3 < w_out; j += 4) {
          __m256 i00 = _mm256_loadu_ps(block_inr0);
          __m256 i01 = _mm256_loadu_ps(block_inr0 + 8);
          __m256 i02 = _mm256_loadu_ps(block_inr0 + 16);
          __m256 i03 = _mm256_loadu_ps(block_inr0 + 24);
          __m256 i04 = _mm256_loadu_ps(block_inr0 + 32);
          __m256 i05 = _mm256_loadu_ps(block_inr0 + 40);
          __m256 i06 = _mm256_loadu_ps(block_inr0 + 48);
          __m256 i07 = _mm256_loadu_ps(block_inr0 + 56);

          __m256 w00 = _mm256_loadu_ps(weights_data);
          __m256 r0 = _mm256_fmadd_ps(i00, w00, _bias);
          __m256 r1 = _mm256_fmadd_ps(i01, w00, _bias);
          __m256 r2 = _mm256_fmadd_ps(i02, w00, _bias);
          __m256 r3 = _mm256_fmadd_ps(i03, w00, _bias);

          __m256 w01 = _mm256_loadu_ps(weights_data + 8);
          r0 = _mm256_fmadd_ps(i01, w01, r0);
          r1 = _mm256_fmadd_ps(i02, w01, r1);
          r2 = _mm256_fmadd_ps(i03, w01, r2);
          r3 = _mm256_fmadd_ps(i04, w01, r3);

          __m256 w02 = _mm256_loadu_ps(weights_data + 16);
          r0 = _mm256_fmadd_ps(i02, w02, r0);
          r1 = _mm256_fmadd_ps(i03, w02, r1);
          r2 = _mm256_fmadd_ps(i04, w02, r2);
          r3 = _mm256_fmadd_ps(i05, w02, r3);

          __m256 w03 = _mm256_loadu_ps(weights_data + 24);
          r0 = _mm256_fmadd_ps(i03, w03, r0);
          r1 = _mm256_fmadd_ps(i04, w03, r1);
          r2 = _mm256_fmadd_ps(i05, w03, r2);
          r3 = _mm256_fmadd_ps(i06, w03, r3);

          __m256 w04 = _mm256_loadu_ps(weights_data + 32);
          r0 = _mm256_fmadd_ps(i04, w04, r0);
          r1 = _mm256_fmadd_ps(i05, w04, r1);
          r2 = _mm256_fmadd_ps(i06, w04, r2);
          r3 = _mm256_fmadd_ps(i07, w04, r3);

          __m256 i10 = _mm256_loadu_ps(block_inr1);
          __m256 i11 = _mm256_loadu_ps(block_inr1 + 8);
          __m256 i12 = _mm256_loadu_ps(block_inr1 + 16);
          __m256 i13 = _mm256_loadu_ps(block_inr1 + 24);
          __m256 i14 = _mm256_loadu_ps(block_inr1 + 32);
          __m256 i15 = _mm256_loadu_ps(block_inr1 + 40);
          __m256 i16 = _mm256_loadu_ps(block_inr1 + 48);
          __m256 i17 = _mm256_loadu_ps(block_inr1 + 56);

          __m256 w10 = _mm256_loadu_ps(weights_data + 40);
          r0 = _mm256_fmadd_ps(i10, w10, r0);
          r1 = _mm256_fmadd_ps(i11, w10, r1);
          r2 = _mm256_fmadd_ps(i12, w10, r2);
          r3 = _mm256_fmadd_ps(i13, w10, r3);

          __m256 w11 = _mm256_loadu_ps(weights_data + 48);
          r0 = _mm256_fmadd_ps(i11, w11, r0);
          r1 = _mm256_fmadd_ps(i12, w11, r1);
          r2 = _mm256_fmadd_ps(i13, w11, r2);
          r3 = _mm256_fmadd_ps(i14, w11, r3);

          __m256 w12 = _mm256_loadu_ps(weights_data + 56);
          r0 = _mm256_fmadd_ps(i12, w12, r0);
          r1 = _mm256_fmadd_ps(i13, w12, r1);
          r2 = _mm256_fmadd_ps(i14, w12, r2);
          r3 = _mm256_fmadd_ps(i15, w12, r3);

          __m256 w13 = _mm256_loadu_ps(weights_data + 64);
          r0 = _mm256_fmadd_ps(i13, w13, r0);
          r1 = _mm256_fmadd_ps(i14, w13, r1);
          r2 = _mm256_fmadd_ps(i15, w13, r2);
          r3 = _mm256_fmadd_ps(i16, w13, r3);

          __m256 w14 = _mm256_loadu_ps(weights_data + 72);
          r0 = _mm256_fmadd_ps(i14, w14, r0);
          r1 = _mm256_fmadd_ps(i15, w14, r1);
          r2 = _mm256_fmadd_ps(i16, w14, r2);
          r3 = _mm256_fmadd_ps(i17, w14, r3);

          __m256 i20 = _mm256_loadu_ps(block_inr2);
          __m256 i21 = _mm256_loadu_ps(block_inr2 + 8);
          __m256 i22 = _mm256_loadu_ps(block_inr2 + 16);
          __m256 i23 = _mm256_loadu_ps(block_inr2 + 24);
          __m256 i24 = _mm256_loadu_ps(block_inr2 + 32);
          __m256 i25 = _mm256_loadu_ps(block_inr2 + 40);
          __m256 i26 = _mm256_loadu_ps(block_inr2 + 48);
          __m256 i27 = _mm256_loadu_ps(block_inr2 + 56);

          __m256 w20 = _mm256_loadu_ps(weights_data + 80);
          r0 = _mm256_fmadd_ps(i20, w20, r0);
          r1 = _mm256_fmadd_ps(i21, w20, r1);
          r2 = _mm256_fmadd_ps(i22, w20, r2);
          r3 = _mm256_fmadd_ps(i23, w20, r3);

          __m256 w21 = _mm256_loadu_ps(weights_data + 88);
          r0 = _mm256_fmadd_ps(i21, w21, r0);
          r1 = _mm256_fmadd_ps(i22, w21, r1);
          r2 = _mm256_fmadd_ps(i23, w21, r2);
          r3 = _mm256_fmadd_ps(i24, w21, r3);

          __m256 w22 = _mm256_loadu_ps(weights_data + 96);
          r0 = _mm256_fmadd_ps(i22, w22, r0);
          r1 = _mm256_fmadd_ps(i23, w22, r1);
          r2 = _mm256_fmadd_ps(i24, w22, r2);
          r3 = _mm256_fmadd_ps(i25, w22, r3);

          __m256 w23 = _mm256_loadu_ps(weights_data + 104);
          r0 = _mm256_fmadd_ps(i23, w23, r0);
          r1 = _mm256_fmadd_ps(i24, w23, r1);
          r2 = _mm256_fmadd_ps(i25, w23, r2);
          r3 = _mm256_fmadd_ps(i26, w23, r3);

          __m256 w24 = _mm256_loadu_ps(weights_data + 112);
          r0 = _mm256_fmadd_ps(i24, w24, r0);
          r1 = _mm256_fmadd_ps(i25, w24, r1);
          r2 = _mm256_fmadd_ps(i26, w24, r2);
          r3 = _mm256_fmadd_ps(i27, w24, r3);

          __m256 i30 = _mm256_loadu_ps(block_inr3);
          __m256 i31 = _mm256_loadu_ps(block_inr3 + 8);
          __m256 i32 = _mm256_loadu_ps(block_inr3 + 16);
          __m256 i33 = _mm256_loadu_ps(block_inr3 + 24);
          __m256 i34 = _mm256_loadu_ps(block_inr3 + 32);
          __m256 i35 = _mm256_loadu_ps(block_inr3 + 40);
          __m256 i36 = _mm256_loadu_ps(block_inr3 + 48);
          __m256 i37 = _mm256_loadu_ps(block_inr3 + 56);

          __m256 w30 = _mm256_loadu_ps(weights_data + 120);
          r0 = _mm256_fmadd_ps(i30, w30, r0);
          r1 = _mm256_fmadd_ps(i31, w30, r1);
          r2 = _mm256_fmadd_ps(i32, w30, r2);
          r3 = _mm256_fmadd_ps(i33, w30, r3);

          __m256 w31 = _mm256_loadu_ps(weights_data + 128);
          r0 = _mm256_fmadd_ps(i31, w31, r0);
          r1 = _mm256_fmadd_ps(i32, w31, r1);
          r2 = _mm256_fmadd_ps(i33, w31, r2);
          r3 = _mm256_fmadd_ps(i34, w31, r3);

          __m256 w32 = _mm256_loadu_ps(weights_data + 136);
          r0 = _mm256_fmadd_ps(i32, w32, r0);
          r1 = _mm256_fmadd_ps(i33, w32, r1);
          r2 = _mm256_fmadd_ps(i34, w32, r2);
          r3 = _mm256_fmadd_ps(i35, w32, r3);

          __m256 w33 = _mm256_loadu_ps(weights_data + 144);
          r0 = _mm256_fmadd_ps(i33, w33, r0);
          r1 = _mm256_fmadd_ps(i34, w33, r1);
          r2 = _mm256_fmadd_ps(i35, w33, r2);
          r3 = _mm256_fmadd_ps(i36, w33, r3);

          __m256 w34 = _mm256_loadu_ps(weights_data + 152);
          r0 = _mm256_fmadd_ps(i34, w34, r0);
          r1 = _mm256_fmadd_ps(i35, w34, r1);
          r2 = _mm256_fmadd_ps(i36, w34, r2);
          r3 = _mm256_fmadd_ps(i37, w34, r3);

          __m256 i40 = _mm256_loadu_ps(block_inr4);
          __m256 i41 = _mm256_loadu_ps(block_inr4 + 8);
          __m256 i42 = _mm256_loadu_ps(block_inr4 + 16);
          __m256 i43 = _mm256_loadu_ps(block_inr4 + 24);
          __m256 i44 = _mm256_loadu_ps(block_inr4 + 32);
          __m256 i45 = _mm256_loadu_ps(block_inr4 + 40);
          __m256 i46 = _mm256_loadu_ps(block_inr4 + 48);
          __m256 i47 = _mm256_loadu_ps(block_inr4 + 56);

          __m256 w40 = _mm256_loadu_ps(weights_data + 160);
          r0 = _mm256_fmadd_ps(i40, w40, r0);
          r1 = _mm256_fmadd_ps(i41, w40, r1);
          r2 = _mm256_fmadd_ps(i42, w40, r2);
          r3 = _mm256_fmadd_ps(i43, w40, r3);

          __m256 w41 = _mm256_loadu_ps(weights_data + 168);
          r0 = _mm256_fmadd_ps(i41, w41, r0);
          r1 = _mm256_fmadd_ps(i42, w41, r1);
          r2 = _mm256_fmadd_ps(i43, w41, r2);
          r3 = _mm256_fmadd_ps(i44, w41, r3);

          __m256 w42 = _mm256_loadu_ps(weights_data + 176);
          r0 = _mm256_fmadd_ps(i42, w42, r0);
          r1 = _mm256_fmadd_ps(i43, w42, r1);
          r2 = _mm256_fmadd_ps(i44, w42, r2);
          r3 = _mm256_fmadd_ps(i45, w42, r3);

          __m256 w43 = _mm256_loadu_ps(weights_data + 184);
          r0 = _mm256_fmadd_ps(i43, w43, r0);
          r1 = _mm256_fmadd_ps(i44, w43, r1);
          r2 = _mm256_fmadd_ps(i45, w43, r2);
          r3 = _mm256_fmadd_ps(i46, w43, r3);

          __m256 w44 = _mm256_loadu_ps(weights_data + 192);
          r0 = _mm256_fmadd_ps(i44, w44, r0);
          r1 = _mm256_fmadd_ps(i45, w44, r1);
          r2 = _mm256_fmadd_ps(i46, w44, r2);
          r3 = _mm256_fmadd_ps(i47, w44, r3);

          __m256 zero = _mm256_setzero_ps();
          if (has_active) {
            if (act_type == lite_api::ActivationType::kRelu) {
              r0 = _mm256_max_ps(r0, zero);
              r1 = _mm256_max_ps(r1, zero);
              r2 = _mm256_max_ps(r2, zero);
              r3 = _mm256_max_ps(r3, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              __m256 six = _mm256_set1_ps(act_param.Relu_clipped_coef);
              r0 = _mm256_min_ps(_mm256_max_ps(r0, zero), six);
              r1 = _mm256_min_ps(_mm256_max_ps(r1, zero), six);
              r2 = _mm256_min_ps(_mm256_max_ps(r2, zero), six);
              r3 = _mm256_min_ps(_mm256_max_ps(r3, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              __m256 negative_slope =
                  _mm256_set1_ps(act_param.Leaky_relu_alpha);
              r0 = _mm256_blendv_ps(r0,
                                    _mm256_mul_ps(negative_slope, r0),
                                    _mm256_cmp_ps(r0, zero, 2));
              r1 = _mm256_blendv_ps(r1,
                                    _mm256_mul_ps(negative_slope, r1),
                                    _mm256_cmp_ps(r1, zero, 2));
              r2 = _mm256_blendv_ps(r2,
                                    _mm256_mul_ps(negative_slope, r2),
                                    _mm256_cmp_ps(r2, zero, 2));
              r3 = _mm256_blendv_ps(r3,
                                    _mm256_mul_ps(negative_slope, r3),
                                    _mm256_cmp_ps(r3, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              __m256 vscale = _mm256_set1_ps(1.0 / act_param.hard_swish_scale);
              __m256 voffset = _mm256_set1_ps(act_param.hard_swish_offset);
              __m256 vthreshold =
                  _mm256_set1_ps(act_param.hard_swish_threshold);
              r0 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r0, voffset))),
                  _mm256_mul_ps(r0, vscale));
              r1 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r1, voffset))),
                  _mm256_mul_ps(r1, vscale));
              r2 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r2, voffset))),
                  _mm256_mul_ps(r2, vscale));
              r3 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r3, voffset))),
                  _mm256_mul_ps(r3, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";
            }
          }

          _mm256_storeu_ps(dout_block, r0);
          _mm256_storeu_ps(dout_block + 8, r1);
          _mm256_storeu_ps(dout_block + 16, r2);
          _mm256_storeu_ps(dout_block + 24, r3);
          dout_block += 32;

          block_inr0 += 4 * block_channel;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        for (; j < w_out; j++) {
          __m256 r = _bias;
          for (int m = 0; m < 5; m++) {
            for (int n = 0; n < 5; n++) {
              __m256 weight = _mm256_loadu_ps(
                  weights_data + 5 * block_channel * m + block_channel * n);
              __m256 input =
                  _mm256_loadu_ps(block_inr0 + block_channel * (j % 4) +
                                  in_len * m + block_channel * n);
              r = _mm256_fmadd_ps(input, weight, r);
            }
          }
          __m256 zero = _mm256_setzero_ps();
          if (has_active) {
            if (act_type == lite_api::ActivationType::kRelu) {
              r = _mm256_max_ps(r, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              __m256 six = _mm256_set1_ps(act_param.Relu_clipped_coef);
              r = _mm256_min_ps(_mm256_max_ps(r, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              __m256 negative_slope =
                  _mm256_set1_ps(act_param.Leaky_relu_alpha);
              r = _mm256_blendv_ps(r,
                                   _mm256_mul_ps(negative_slope, r),
                                   _mm256_cmp_ps(r, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              __m256 vscale = _mm256_set1_ps(1.0 / act_param.hard_swish_scale);
              __m256 voffset = _mm256_set1_ps(act_param.hard_swish_offset);
              __m256 vthreshold =
                  _mm256_set1_ps(act_param.hard_swish_threshold);
              r = _mm256_mul_ps(
                  _mm256_min_ps(vthreshold,
                                _mm256_max_ps(zero, _mm256_add_ps(r, voffset))),
                  _mm256_mul_ps(r, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";
            }
          }
          _mm256_storeu_ps(dout_block, r);
          dout_block += 8;
        }
      }

      unpackC8_with_Cleft(
          pack_out, dout_ptr, size_out_channel, real_block_channel);
    }
  }

  TargetFree(TARGET(kX86), pack_weight);
  TargetFree(TARGET(kX86), pack_input);
  TargetFree(TARGET(kX86), pack_out);
}
void conv_depthwise_5x5s2(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          bool flag_bias,
                          const operators::ActivationParam act_param) {
  bool has_active = act_param.has_active;
  auto act_type = act_param.active_type;

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int block_channel = 8;
  int in_len = block_channel * (2 * pad + w_in);

  int channel_num = ROUNDUP(ch_in, block_channel);
  float* pack_weight = static_cast<float*>(TargetMalloc(
      TARGET(kX86),
      1 * channel_num / block_channel * 5 * 5 * block_channel * sizeof(float)));
  float* pack_input = static_cast<float*>(TargetMalloc(
      TARGET(kX86),
      1 * (h_in + 2 * pad) * (w_in + 2 * pad) * block_channel * sizeof(float)));
  float* pack_out = static_cast<float*>(TargetMalloc(
      TARGET(kX86), 1 * h_out * w_out * block_channel * sizeof(float)));

  packC8_with_Cleft(weights, pack_weight, {0, 0, 0, 0}, 5, 5, ch_in);

  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_out * size_out_channel;

    for (int c = 0; c < ch_out; c += block_channel) {
      int real_block_channel = Min(block_channel, ch_out - c);
      auto* dout_ptr = dout_batch + c * size_out_channel;
      auto* din_ptr = din_batch + c * size_in_channel;
      auto* weights_data = pack_weight + c * 5 * 5;
      packC8_with_Cleft(din_ptr,
                        pack_input,
                        {pad, pad, pad, pad},
                        h_in,
                        w_in,
                        real_block_channel);
      int mask_ptr[8] = {0};
      for (int i = 0; i < 8; i++) {
        if (real_block_channel > i) {
          mask_ptr[i] = 0x80000000;
        }
      }
      __m256i bias_mask = _mm256_set_epi32(mask_ptr[7],
                                           mask_ptr[6],
                                           mask_ptr[5],
                                           mask_ptr[4],
                                           mask_ptr[3],
                                           mask_ptr[2],
                                           mask_ptr[1],
                                           mask_ptr[0]);
      __m256 _bias = flag_bias ? _mm256_maskload_ps(bias + c, bias_mask)
                               : _mm256_set1_ps(0.f);

      for (int i = 0; i < h_out; i++) {
        const float* block_inr0 = pack_input + i * 2 * in_len;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        int j = 0;
        float* dout_block = pack_out + i * w_out * block_channel;
        for (; j + 3 < w_out; j += 4) {
          __m256 i00 = _mm256_loadu_ps(block_inr0);
          __m256 i01 = _mm256_loadu_ps(block_inr0 + 8);
          __m256 i02 = _mm256_loadu_ps(block_inr0 + 16);
          __m256 i03 = _mm256_loadu_ps(block_inr0 + 24);
          __m256 i04 = _mm256_loadu_ps(block_inr0 + 32);
          __m256 i05 = _mm256_loadu_ps(block_inr0 + 40);
          __m256 i06 = _mm256_loadu_ps(block_inr0 + 48);
          __m256 i07 = _mm256_loadu_ps(block_inr0 + 56);
          __m256 i08 = _mm256_loadu_ps(block_inr0 + 64);
          __m256 i09 = _mm256_loadu_ps(block_inr0 + 72);
          __m256 i0a = _mm256_loadu_ps(block_inr0 + 80);

          __m256 w00 = _mm256_loadu_ps(weights_data);
          __m256 r0 = _mm256_fmadd_ps(i00, w00, _bias);
          __m256 r1 = _mm256_fmadd_ps(i02, w00, _bias);
          __m256 r2 = _mm256_fmadd_ps(i04, w00, _bias);
          __m256 r3 = _mm256_fmadd_ps(i06, w00, _bias);

          __m256 w01 = _mm256_loadu_ps(weights_data + 8);
          r0 = _mm256_fmadd_ps(i01, w01, r0);
          r1 = _mm256_fmadd_ps(i03, w01, r1);
          r2 = _mm256_fmadd_ps(i05, w01, r2);
          r3 = _mm256_fmadd_ps(i07, w01, r3);

          __m256 w02 = _mm256_loadu_ps(weights_data + 16);
          r0 = _mm256_fmadd_ps(i02, w02, r0);
          r1 = _mm256_fmadd_ps(i04, w02, r1);
          r2 = _mm256_fmadd_ps(i06, w02, r2);
          r3 = _mm256_fmadd_ps(i08, w02, r3);

          __m256 w03 = _mm256_loadu_ps(weights_data + 24);
          r0 = _mm256_fmadd_ps(i03, w03, r0);
          r1 = _mm256_fmadd_ps(i05, w03, r1);
          r2 = _mm256_fmadd_ps(i07, w03, r2);
          r3 = _mm256_fmadd_ps(i09, w03, r3);

          __m256 w04 = _mm256_loadu_ps(weights_data + 32);
          r0 = _mm256_fmadd_ps(i04, w04, r0);
          r1 = _mm256_fmadd_ps(i06, w04, r1);
          r2 = _mm256_fmadd_ps(i08, w04, r2);
          r3 = _mm256_fmadd_ps(i0a, w04, r3);

          __m256 i10 = _mm256_loadu_ps(block_inr1);
          __m256 i11 = _mm256_loadu_ps(block_inr1 + 8);
          __m256 i12 = _mm256_loadu_ps(block_inr1 + 16);
          __m256 i13 = _mm256_loadu_ps(block_inr1 + 24);
          __m256 i14 = _mm256_loadu_ps(block_inr1 + 32);
          __m256 i15 = _mm256_loadu_ps(block_inr1 + 40);
          __m256 i16 = _mm256_loadu_ps(block_inr1 + 48);
          __m256 i17 = _mm256_loadu_ps(block_inr1 + 56);
          __m256 i18 = _mm256_loadu_ps(block_inr1 + 64);
          __m256 i19 = _mm256_loadu_ps(block_inr1 + 72);
          __m256 i1a = _mm256_loadu_ps(block_inr1 + 80);

          __m256 w10 = _mm256_loadu_ps(weights_data + 40);
          r0 = _mm256_fmadd_ps(i10, w10, r0);
          r1 = _mm256_fmadd_ps(i12, w10, r1);
          r2 = _mm256_fmadd_ps(i14, w10, r2);
          r3 = _mm256_fmadd_ps(i16, w10, r3);

          __m256 w11 = _mm256_loadu_ps(weights_data + 48);
          r0 = _mm256_fmadd_ps(i11, w11, r0);
          r1 = _mm256_fmadd_ps(i13, w11, r1);
          r2 = _mm256_fmadd_ps(i15, w11, r2);
          r3 = _mm256_fmadd_ps(i17, w11, r3);

          __m256 w12 = _mm256_loadu_ps(weights_data + 56);
          r0 = _mm256_fmadd_ps(i12, w12, r0);
          r1 = _mm256_fmadd_ps(i14, w12, r1);
          r2 = _mm256_fmadd_ps(i16, w12, r2);
          r3 = _mm256_fmadd_ps(i18, w12, r3);

          __m256 w13 = _mm256_loadu_ps(weights_data + 64);
          r0 = _mm256_fmadd_ps(i13, w13, r0);
          r1 = _mm256_fmadd_ps(i15, w13, r1);
          r2 = _mm256_fmadd_ps(i17, w13, r2);
          r3 = _mm256_fmadd_ps(i19, w13, r3);

          __m256 w14 = _mm256_loadu_ps(weights_data + 72);
          r0 = _mm256_fmadd_ps(i14, w14, r0);
          r1 = _mm256_fmadd_ps(i16, w14, r1);
          r2 = _mm256_fmadd_ps(i18, w14, r2);
          r3 = _mm256_fmadd_ps(i1a, w14, r3);

          __m256 i20 = _mm256_loadu_ps(block_inr2);
          __m256 i21 = _mm256_loadu_ps(block_inr2 + 8);
          __m256 i22 = _mm256_loadu_ps(block_inr2 + 16);
          __m256 i23 = _mm256_loadu_ps(block_inr2 + 24);
          __m256 i24 = _mm256_loadu_ps(block_inr2 + 32);
          __m256 i25 = _mm256_loadu_ps(block_inr2 + 40);
          __m256 i26 = _mm256_loadu_ps(block_inr2 + 48);
          __m256 i27 = _mm256_loadu_ps(block_inr2 + 56);
          __m256 i28 = _mm256_loadu_ps(block_inr2 + 64);
          __m256 i29 = _mm256_loadu_ps(block_inr2 + 72);
          __m256 i2a = _mm256_loadu_ps(block_inr2 + 80);

          __m256 w20 = _mm256_loadu_ps(weights_data + 80);
          r0 = _mm256_fmadd_ps(i20, w20, r0);
          r1 = _mm256_fmadd_ps(i22, w20, r1);
          r2 = _mm256_fmadd_ps(i24, w20, r2);
          r3 = _mm256_fmadd_ps(i26, w20, r3);

          __m256 w21 = _mm256_loadu_ps(weights_data + 88);
          r0 = _mm256_fmadd_ps(i21, w21, r0);
          r1 = _mm256_fmadd_ps(i23, w21, r1);
          r2 = _mm256_fmadd_ps(i25, w21, r2);
          r3 = _mm256_fmadd_ps(i27, w21, r3);

          __m256 w22 = _mm256_loadu_ps(weights_data + 96);
          r0 = _mm256_fmadd_ps(i22, w22, r0);
          r1 = _mm256_fmadd_ps(i24, w22, r1);
          r2 = _mm256_fmadd_ps(i26, w22, r2);
          r3 = _mm256_fmadd_ps(i28, w22, r3);

          __m256 w23 = _mm256_loadu_ps(weights_data + 104);
          r0 = _mm256_fmadd_ps(i23, w23, r0);
          r1 = _mm256_fmadd_ps(i25, w23, r1);
          r2 = _mm256_fmadd_ps(i27, w23, r2);
          r3 = _mm256_fmadd_ps(i29, w23, r3);

          __m256 w24 = _mm256_loadu_ps(weights_data + 112);
          r0 = _mm256_fmadd_ps(i24, w24, r0);
          r1 = _mm256_fmadd_ps(i26, w24, r1);
          r2 = _mm256_fmadd_ps(i28, w24, r2);
          r3 = _mm256_fmadd_ps(i2a, w24, r3);

          __m256 i30 = _mm256_loadu_ps(block_inr3);
          __m256 i31 = _mm256_loadu_ps(block_inr3 + 8);
          __m256 i32 = _mm256_loadu_ps(block_inr3 + 16);
          __m256 i33 = _mm256_loadu_ps(block_inr3 + 24);
          __m256 i34 = _mm256_loadu_ps(block_inr3 + 32);
          __m256 i35 = _mm256_loadu_ps(block_inr3 + 40);
          __m256 i36 = _mm256_loadu_ps(block_inr3 + 48);
          __m256 i37 = _mm256_loadu_ps(block_inr3 + 56);
          __m256 i38 = _mm256_loadu_ps(block_inr3 + 64);
          __m256 i39 = _mm256_loadu_ps(block_inr3 + 72);
          __m256 i3a = _mm256_loadu_ps(block_inr3 + 80);

          __m256 w30 = _mm256_loadu_ps(weights_data + 120);
          r0 = _mm256_fmadd_ps(i30, w30, r0);
          r1 = _mm256_fmadd_ps(i32, w30, r1);
          r2 = _mm256_fmadd_ps(i34, w30, r2);
          r3 = _mm256_fmadd_ps(i36, w30, r3);

          __m256 w31 = _mm256_loadu_ps(weights_data + 128);
          r0 = _mm256_fmadd_ps(i31, w31, r0);
          r1 = _mm256_fmadd_ps(i33, w31, r1);
          r2 = _mm256_fmadd_ps(i35, w31, r2);
          r3 = _mm256_fmadd_ps(i37, w31, r3);

          __m256 w32 = _mm256_loadu_ps(weights_data + 136);
          r0 = _mm256_fmadd_ps(i32, w32, r0);
          r1 = _mm256_fmadd_ps(i34, w32, r1);
          r2 = _mm256_fmadd_ps(i36, w32, r2);
          r3 = _mm256_fmadd_ps(i38, w32, r3);

          __m256 w33 = _mm256_loadu_ps(weights_data + 144);
          r0 = _mm256_fmadd_ps(i33, w33, r0);
          r1 = _mm256_fmadd_ps(i35, w33, r1);
          r2 = _mm256_fmadd_ps(i37, w33, r2);
          r3 = _mm256_fmadd_ps(i39, w33, r3);

          __m256 w34 = _mm256_loadu_ps(weights_data + 152);
          r0 = _mm256_fmadd_ps(i34, w34, r0);
          r1 = _mm256_fmadd_ps(i36, w34, r1);
          r2 = _mm256_fmadd_ps(i38, w34, r2);
          r3 = _mm256_fmadd_ps(i3a, w34, r3);

          __m256 i40 = _mm256_loadu_ps(block_inr4);
          __m256 i41 = _mm256_loadu_ps(block_inr4 + 8);
          __m256 i42 = _mm256_loadu_ps(block_inr4 + 16);
          __m256 i43 = _mm256_loadu_ps(block_inr4 + 24);
          __m256 i44 = _mm256_loadu_ps(block_inr4 + 32);
          __m256 i45 = _mm256_loadu_ps(block_inr4 + 40);
          __m256 i46 = _mm256_loadu_ps(block_inr4 + 48);
          __m256 i47 = _mm256_loadu_ps(block_inr4 + 56);
          __m256 i48 = _mm256_loadu_ps(block_inr4 + 64);
          __m256 i49 = _mm256_loadu_ps(block_inr4 + 72);
          __m256 i4a = _mm256_loadu_ps(block_inr4 + 80);

          __m256 w40 = _mm256_loadu_ps(weights_data + 160);
          r0 = _mm256_fmadd_ps(i40, w40, r0);
          r1 = _mm256_fmadd_ps(i42, w40, r1);
          r2 = _mm256_fmadd_ps(i44, w40, r2);
          r3 = _mm256_fmadd_ps(i46, w40, r3);

          __m256 w41 = _mm256_loadu_ps(weights_data + 168);
          r0 = _mm256_fmadd_ps(i41, w41, r0);
          r1 = _mm256_fmadd_ps(i43, w41, r1);
          r2 = _mm256_fmadd_ps(i45, w41, r2);
          r3 = _mm256_fmadd_ps(i47, w41, r3);

          __m256 w42 = _mm256_loadu_ps(weights_data + 176);
          r0 = _mm256_fmadd_ps(i42, w42, r0);
          r1 = _mm256_fmadd_ps(i44, w42, r1);
          r2 = _mm256_fmadd_ps(i46, w42, r2);
          r3 = _mm256_fmadd_ps(i48, w42, r3);

          __m256 w43 = _mm256_loadu_ps(weights_data + 184);
          r0 = _mm256_fmadd_ps(i43, w43, r0);
          r1 = _mm256_fmadd_ps(i45, w43, r1);
          r2 = _mm256_fmadd_ps(i47, w43, r2);
          r3 = _mm256_fmadd_ps(i49, w43, r3);

          __m256 w44 = _mm256_loadu_ps(weights_data + 192);
          r0 = _mm256_fmadd_ps(i44, w44, r0);
          r1 = _mm256_fmadd_ps(i46, w44, r1);
          r2 = _mm256_fmadd_ps(i48, w44, r2);
          r3 = _mm256_fmadd_ps(i4a, w44, r3);

          __m256 zero = _mm256_setzero_ps();
          if (has_active) {  // process activation
            if (act_type == lite_api::ActivationType::kRelu) {
              r0 = _mm256_max_ps(r0, zero);
              r1 = _mm256_max_ps(r1, zero);
              r2 = _mm256_max_ps(r2, zero);
              r3 = _mm256_max_ps(r3, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              __m256 six = _mm256_set1_ps(act_param.Relu_clipped_coef);
              r0 = _mm256_min_ps(_mm256_max_ps(r0, zero), six);
              r1 = _mm256_min_ps(_mm256_max_ps(r1, zero), six);
              r2 = _mm256_min_ps(_mm256_max_ps(r2, zero), six);
              r3 = _mm256_min_ps(_mm256_max_ps(r3, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              __m256 negative_slope =
                  _mm256_set1_ps(act_param.Leaky_relu_alpha);
              r0 = _mm256_blendv_ps(r0,
                                    _mm256_mul_ps(negative_slope, r0),
                                    _mm256_cmp_ps(r0, zero, 2));
              r1 = _mm256_blendv_ps(r1,
                                    _mm256_mul_ps(negative_slope, r1),
                                    _mm256_cmp_ps(r1, zero, 2));
              r2 = _mm256_blendv_ps(r2,
                                    _mm256_mul_ps(negative_slope, r2),
                                    _mm256_cmp_ps(r2, zero, 2));
              r3 = _mm256_blendv_ps(r3,
                                    _mm256_mul_ps(negative_slope, r3),
                                    _mm256_cmp_ps(r3, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              __m256 vscale = _mm256_set1_ps(1.0 / act_param.hard_swish_scale);
              __m256 voffset = _mm256_set1_ps(act_param.hard_swish_offset);
              __m256 vthreshold =
                  _mm256_set1_ps(act_param.hard_swish_threshold);
              r0 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r0, voffset))),
                  _mm256_mul_ps(r0, vscale));
              r1 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r1, voffset))),
                  _mm256_mul_ps(r1, vscale));
              r2 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r2, voffset))),
                  _mm256_mul_ps(r2, vscale));
              r3 = _mm256_mul_ps(
                  _mm256_min_ps(
                      vthreshold,
                      _mm256_max_ps(zero, _mm256_add_ps(r3, voffset))),
                  _mm256_mul_ps(r3, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";
            }
          }

          _mm256_storeu_ps(dout_block, r0);
          _mm256_storeu_ps(dout_block + 8, r1);
          _mm256_storeu_ps(dout_block + 16, r2);
          _mm256_storeu_ps(dout_block + 24, r3);
          dout_block += 32;

          block_inr0 += 8 * block_channel;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        for (; j < w_out; j++) {
          __m256 r = _bias;
          for (int m = 0; m < 5; m++) {
            for (int n = 0; n < 5; n++) {
              __m256 weight = _mm256_loadu_ps(
                  weights_data + 5 * block_channel * m + block_channel * n);
              __m256 input =
                  _mm256_loadu_ps(block_inr0 + block_channel * (j % 4) * 2 +
                                  in_len * m + block_channel * n);
              r = _mm256_fmadd_ps(input, weight, r);
            }
          }
          __m256 zero = _mm256_setzero_ps();
          if (has_active) {  // process activation
            if (act_type == lite_api::ActivationType::kRelu) {
              r = _mm256_max_ps(r, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              __m256 six = _mm256_set1_ps(act_param.Relu_clipped_coef);
              r = _mm256_min_ps(_mm256_max_ps(r, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              __m256 negative_slope =
                  _mm256_set1_ps(act_param.Leaky_relu_alpha);
              r = _mm256_blendv_ps(r,
                                   _mm256_mul_ps(negative_slope, r),
                                   _mm256_cmp_ps(r, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              __m256 vscale = _mm256_set1_ps(1.0 / act_param.hard_swish_scale);
              __m256 voffset = _mm256_set1_ps(act_param.hard_swish_offset);
              __m256 vthreshold =
                  _mm256_set1_ps(act_param.hard_swish_threshold);
              r = _mm256_mul_ps(
                  _mm256_min_ps(vthreshold,
                                _mm256_max_ps(zero, _mm256_add_ps(r, voffset))),
                  _mm256_mul_ps(r, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << "not supported";
            }
          }
          _mm256_storeu_ps(dout_block, r);
          dout_block += 8;
        }
      }

      unpackC8_with_Cleft(
          pack_out, dout_ptr, size_out_channel, real_block_channel);
    }
  }

  TargetFree(TARGET(kX86), pack_weight);
  TargetFree(TARGET(kX86), pack_input);
  TargetFree(TARGET(kX86), pack_out);
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
