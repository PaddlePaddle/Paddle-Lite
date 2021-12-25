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

#include "lite/backends/x86/math/conv_depthwise_5x5.h"
#include <vector>
#include "lite/backends/x86/math/avx/avx_mathfuns.h"
#include "lite/backends/x86/math/avx/avx_mathfuns.h"
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/backends/x86/math/conv_depthwise_impl.h"
#include "lite/backends/x86/math/sse/conv_utils.h"
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
  int in_len = block_channel * (2 * pad + w_in);

  int channel_num = ROUNDUP(ch_in, block_channel);
  float* pack_weight = static_cast<float*>(
      TargetMalloc(TARGET(kX86), channel_num * 5 * 5 * sizeof(float)));
  float* pack_input = static_cast<float*>(TargetMalloc(
      TARGET(kX86),
      (h_in + 2 * pad) * (w_in + 2 * pad) * block_channel * sizeof(float)));
  float* pack_out = static_cast<float*>(TargetMalloc(
      TARGET(kX86), h_out * w_out * block_channel * sizeof(float)));

#ifdef __AVX__
  packC8_common(weights, pack_weight, {0, 0, 0, 0}, 5, 5, ch_in);
#else
  packC4_common(weights, pack_weight, {0, 0, 0, 0}, 5, 5, ch_in);
#endif

  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_out * size_out_channel;

    for (int c = 0; c < ch_out; c += block_channel) {
      int real_block_channel = Min(block_channel, ch_out - c);
      auto* dout_ptr = dout_batch + c * size_out_channel;
      auto* din_ptr = din_batch + c * size_in_channel;
      auto* weights_data = pack_weight + c * 5 * 5;

#ifdef __AVX__
      packC8_common(din_ptr,
                    pack_input,
                    {pad, pad, pad, pad},
                    h_in,
                    w_in,
                    real_block_channel);
#else
      packC4_common(din_ptr,
                    pack_input,
                    {pad, pad, pad, pad},
                    h_in,
                    w_in,
                    real_block_channel);
#endif

      float bias_ptr[block_channel] = {0.f};
      if (flag_bias) {
        for (int i = 0; i < block_channel; i++) {
          if (real_block_channel > i) {
            bias_ptr[i] = *(bias + c + i);
          }
        }
      }

      Type _bias = loadu_ps(bias_ptr);

      for (int i = 0; i < h_out; i++) {
        const float* block_inr0 = pack_input + i * in_len;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        int j = 0;
        float* dout_block = pack_out + i * w_out * block_channel;
        for (; j + 3 < w_out; j += 4) {
          Type i00 = loadu_ps(block_inr0);
          Type i01 = loadu_ps(block_inr0 + 1 * block_channel);
          Type i02 = loadu_ps(block_inr0 + 2 * block_channel);
          Type i03 = loadu_ps(block_inr0 + 3 * block_channel);
          Type i04 = loadu_ps(block_inr0 + 4 * block_channel);
          Type i05 = loadu_ps(block_inr0 + 5 * block_channel);
          Type i06 = loadu_ps(block_inr0 + 6 * block_channel);
          Type i07 = loadu_ps(block_inr0 + 7 * block_channel);

          Type w00 = loadu_ps(weights_data);
          Type r0 = fmadd_ps(i00, w00, _bias);
          Type r1 = fmadd_ps(i01, w00, _bias);
          Type r2 = fmadd_ps(i02, w00, _bias);
          Type r3 = fmadd_ps(i03, w00, _bias);

          Type w01 = loadu_ps(weights_data + block_channel);
          r0 = fmadd_ps(i01, w01, r0);
          r1 = fmadd_ps(i02, w01, r1);
          r2 = fmadd_ps(i03, w01, r2);
          r3 = fmadd_ps(i04, w01, r3);

          Type w02 = loadu_ps(weights_data + 2 * block_channel);
          r0 = fmadd_ps(i02, w02, r0);
          r1 = fmadd_ps(i03, w02, r1);
          r2 = fmadd_ps(i04, w02, r2);
          r3 = fmadd_ps(i05, w02, r3);

          Type w03 = loadu_ps(weights_data + 3 * block_channel);
          r0 = fmadd_ps(i03, w03, r0);
          r1 = fmadd_ps(i04, w03, r1);
          r2 = fmadd_ps(i05, w03, r2);
          r3 = fmadd_ps(i06, w03, r3);

          Type w04 = loadu_ps(weights_data + 4 * block_channel);
          r0 = fmadd_ps(i04, w04, r0);
          r1 = fmadd_ps(i05, w04, r1);
          r2 = fmadd_ps(i06, w04, r2);
          r3 = fmadd_ps(i07, w04, r3);

          Type i10 = loadu_ps(block_inr1);
          Type i11 = loadu_ps(block_inr1 + 1 * block_channel);
          Type i12 = loadu_ps(block_inr1 + 2 * block_channel);
          Type i13 = loadu_ps(block_inr1 + 3 * block_channel);
          Type i14 = loadu_ps(block_inr1 + 4 * block_channel);
          Type i15 = loadu_ps(block_inr1 + 5 * block_channel);
          Type i16 = loadu_ps(block_inr1 + 6 * block_channel);
          Type i17 = loadu_ps(block_inr1 + 7 * block_channel);

          Type w10 = loadu_ps(weights_data + 5 * block_channel);
          r0 = fmadd_ps(i10, w10, r0);
          r1 = fmadd_ps(i11, w10, r1);
          r2 = fmadd_ps(i12, w10, r2);
          r3 = fmadd_ps(i13, w10, r3);

          Type w11 = loadu_ps(weights_data + 6 * block_channel);
          r0 = fmadd_ps(i11, w11, r0);
          r1 = fmadd_ps(i12, w11, r1);
          r2 = fmadd_ps(i13, w11, r2);
          r3 = fmadd_ps(i14, w11, r3);

          Type w12 = loadu_ps(weights_data + 7 * block_channel);
          r0 = fmadd_ps(i12, w12, r0);
          r1 = fmadd_ps(i13, w12, r1);
          r2 = fmadd_ps(i14, w12, r2);
          r3 = fmadd_ps(i15, w12, r3);

          Type w13 = loadu_ps(weights_data + 8 * block_channel);
          r0 = fmadd_ps(i13, w13, r0);
          r1 = fmadd_ps(i14, w13, r1);
          r2 = fmadd_ps(i15, w13, r2);
          r3 = fmadd_ps(i16, w13, r3);

          Type w14 = loadu_ps(weights_data + 9 * block_channel);
          r0 = fmadd_ps(i14, w14, r0);
          r1 = fmadd_ps(i15, w14, r1);
          r2 = fmadd_ps(i16, w14, r2);
          r3 = fmadd_ps(i17, w14, r3);

          Type i20 = loadu_ps(block_inr2);
          Type i21 = loadu_ps(block_inr2 + 1 * block_channel);
          Type i22 = loadu_ps(block_inr2 + 2 * block_channel);
          Type i23 = loadu_ps(block_inr2 + 3 * block_channel);
          Type i24 = loadu_ps(block_inr2 + 4 * block_channel);
          Type i25 = loadu_ps(block_inr2 + 5 * block_channel);
          Type i26 = loadu_ps(block_inr2 + 6 * block_channel);
          Type i27 = loadu_ps(block_inr2 + 7 * block_channel);

          Type w20 = loadu_ps(weights_data + 10 * block_channel);
          r0 = fmadd_ps(i20, w20, r0);
          r1 = fmadd_ps(i21, w20, r1);
          r2 = fmadd_ps(i22, w20, r2);
          r3 = fmadd_ps(i23, w20, r3);

          Type w21 = loadu_ps(weights_data + 11 * block_channel);
          r0 = fmadd_ps(i21, w21, r0);
          r1 = fmadd_ps(i22, w21, r1);
          r2 = fmadd_ps(i23, w21, r2);
          r3 = fmadd_ps(i24, w21, r3);

          Type w22 = loadu_ps(weights_data + 12 * block_channel);
          r0 = fmadd_ps(i22, w22, r0);
          r1 = fmadd_ps(i23, w22, r1);
          r2 = fmadd_ps(i24, w22, r2);
          r3 = fmadd_ps(i25, w22, r3);

          Type w23 = loadu_ps(weights_data + 13 * block_channel);
          r0 = fmadd_ps(i23, w23, r0);
          r1 = fmadd_ps(i24, w23, r1);
          r2 = fmadd_ps(i25, w23, r2);
          r3 = fmadd_ps(i26, w23, r3);

          Type w24 = loadu_ps(weights_data + 14 * block_channel);
          r0 = fmadd_ps(i24, w24, r0);
          r1 = fmadd_ps(i25, w24, r1);
          r2 = fmadd_ps(i26, w24, r2);
          r3 = fmadd_ps(i27, w24, r3);

          Type i30 = loadu_ps(block_inr3);
          Type i31 = loadu_ps(block_inr3 + 1 * block_channel);
          Type i32 = loadu_ps(block_inr3 + 2 * block_channel);
          Type i33 = loadu_ps(block_inr3 + 3 * block_channel);
          Type i34 = loadu_ps(block_inr3 + 4 * block_channel);
          Type i35 = loadu_ps(block_inr3 + 5 * block_channel);
          Type i36 = loadu_ps(block_inr3 + 6 * block_channel);
          Type i37 = loadu_ps(block_inr3 + 7 * block_channel);

          Type w30 = loadu_ps(weights_data + 15 * block_channel);
          r0 = fmadd_ps(i30, w30, r0);
          r1 = fmadd_ps(i31, w30, r1);
          r2 = fmadd_ps(i32, w30, r2);
          r3 = fmadd_ps(i33, w30, r3);

          Type w31 = loadu_ps(weights_data + 16 * block_channel);
          r0 = fmadd_ps(i31, w31, r0);
          r1 = fmadd_ps(i32, w31, r1);
          r2 = fmadd_ps(i33, w31, r2);
          r3 = fmadd_ps(i34, w31, r3);

          Type w32 = loadu_ps(weights_data + 17 * block_channel);
          r0 = fmadd_ps(i32, w32, r0);
          r1 = fmadd_ps(i33, w32, r1);
          r2 = fmadd_ps(i34, w32, r2);
          r3 = fmadd_ps(i35, w32, r3);

          Type w33 = loadu_ps(weights_data + 18 * block_channel);
          r0 = fmadd_ps(i33, w33, r0);
          r1 = fmadd_ps(i34, w33, r1);
          r2 = fmadd_ps(i35, w33, r2);
          r3 = fmadd_ps(i36, w33, r3);

          Type w34 = loadu_ps(weights_data + 19 * block_channel);
          r0 = fmadd_ps(i34, w34, r0);
          r1 = fmadd_ps(i35, w34, r1);
          r2 = fmadd_ps(i36, w34, r2);
          r3 = fmadd_ps(i37, w34, r3);

          Type i40 = loadu_ps(block_inr4);
          Type i41 = loadu_ps(block_inr4 + 1 * block_channel);
          Type i42 = loadu_ps(block_inr4 + 2 * block_channel);
          Type i43 = loadu_ps(block_inr4 + 3 * block_channel);
          Type i44 = loadu_ps(block_inr4 + 4 * block_channel);
          Type i45 = loadu_ps(block_inr4 + 5 * block_channel);
          Type i46 = loadu_ps(block_inr4 + 6 * block_channel);
          Type i47 = loadu_ps(block_inr4 + 7 * block_channel);

          Type w40 = loadu_ps(weights_data + 20 * block_channel);
          r0 = fmadd_ps(i40, w40, r0);
          r1 = fmadd_ps(i41, w40, r1);
          r2 = fmadd_ps(i42, w40, r2);
          r3 = fmadd_ps(i43, w40, r3);

          Type w41 = loadu_ps(weights_data + 21 * block_channel);
          r0 = fmadd_ps(i41, w41, r0);
          r1 = fmadd_ps(i42, w41, r1);
          r2 = fmadd_ps(i43, w41, r2);
          r3 = fmadd_ps(i44, w41, r3);

          Type w42 = loadu_ps(weights_data + 22 * block_channel);
          r0 = fmadd_ps(i42, w42, r0);
          r1 = fmadd_ps(i43, w42, r1);
          r2 = fmadd_ps(i44, w42, r2);
          r3 = fmadd_ps(i45, w42, r3);

          Type w43 = loadu_ps(weights_data + 23 * block_channel);
          r0 = fmadd_ps(i43, w43, r0);
          r1 = fmadd_ps(i44, w43, r1);
          r2 = fmadd_ps(i45, w43, r2);
          r3 = fmadd_ps(i46, w43, r3);

          Type w44 = loadu_ps(weights_data + 24 * block_channel);
          r0 = fmadd_ps(i44, w44, r0);
          r1 = fmadd_ps(i45, w44, r1);
          r2 = fmadd_ps(i46, w44, r2);
          r3 = fmadd_ps(i47, w44, r3);

          Type zero = setzero_ps();
          if (has_active) {
            if (act_type == lite_api::ActivationType::kRelu) {
              r0 = max_ps(r0, zero);
              r1 = max_ps(r1, zero);
              r2 = max_ps(r2, zero);
              r3 = max_ps(r3, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              Type six = set1_ps(act_param.Relu_clipped_coef);
              r0 = min_ps(max_ps(r0, zero), six);
              r1 = min_ps(max_ps(r1, zero), six);
              r2 = min_ps(max_ps(r2, zero), six);
              r3 = min_ps(max_ps(r3, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r0 = blendv_ps(
                  r0, mul_ps(negative_slope, r0), cmp_ps(r0, zero, 2));
              r1 = blendv_ps(
                  r1, mul_ps(negative_slope, r1), cmp_ps(r1, zero, 2));
              r2 = blendv_ps(
                  r2, mul_ps(negative_slope, r2), cmp_ps(r2, zero, 2));
              r3 = blendv_ps(
                  r3, mul_ps(negative_slope, r3), cmp_ps(r3, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              Type vscale = set1_ps(1.0 / act_param.hard_swish_scale);
              Type voffset = set1_ps(act_param.hard_swish_offset);
              Type vthreshold = set1_ps(act_param.hard_swish_threshold);
              r0 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r0, voffset))),
                          mul_ps(r0, vscale));
              r1 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r1, voffset))),
                          mul_ps(r1, vscale));
              r2 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r2, voffset))),
                          mul_ps(r2, vscale));
              r3 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r3, voffset))),
                          mul_ps(r3, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";
            }
          }

          storeu_ps(dout_block, r0);
          storeu_ps(dout_block + block_channel, r1);
          storeu_ps(dout_block + 2 * block_channel, r2);
          storeu_ps(dout_block + 3 * block_channel, r3);
          dout_block += 4 * block_channel;

          block_inr0 += 4 * block_channel;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        for (; j < w_out; j++) {
          Type r = _bias;
          for (int m = 0; m < 5; m++) {
            for (int n = 0; n < 5; n++) {
              Type weight = loadu_ps(weights_data + 5 * block_channel * m +
                                     block_channel * n);
              Type input = loadu_ps(block_inr0 + block_channel * (j % 4) +
                                    in_len * m + block_channel * n);
              r = fmadd_ps(input, weight, r);
            }
          }
          Type zero = setzero_ps();
          if (has_active) {
            if (act_type == lite_api::ActivationType::kRelu) {
              r = max_ps(r, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              Type six = set1_ps(act_param.Relu_clipped_coef);
              r = min_ps(max_ps(r, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r = blendv_ps(r, mul_ps(negative_slope, r), cmp_ps(r, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              Type vscale = set1_ps(1.0 / act_param.hard_swish_scale);
              Type voffset = set1_ps(act_param.hard_swish_offset);
              Type vthreshold = set1_ps(act_param.hard_swish_threshold);
              r = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r, voffset))),
                         mul_ps(r, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";
            }
          }
          storeu_ps(dout_block, r);
          dout_block += block_channel;
        }
      }

#ifdef __AVX__
      unpackC8_common(pack_out, dout_ptr, size_out_channel, real_block_channel);
#else
      unpackC4_common(pack_out, dout_ptr, size_out_channel, real_block_channel);
#endif
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
  int in_len = block_channel * (2 * pad + w_in);

  int channel_num = ROUNDUP(ch_in, block_channel);
  float* pack_weight = static_cast<float*>(
      TargetMalloc(TARGET(kX86), channel_num * 5 * 5 * sizeof(float)));
  float* pack_input = static_cast<float*>(TargetMalloc(
      TARGET(kX86),
      (h_in + 2 * pad) * (w_in + 2 * pad) * block_channel * sizeof(float)));
  float* pack_out = static_cast<float*>(TargetMalloc(
      TARGET(kX86), h_out * w_out * block_channel * sizeof(float)));

#ifdef __AVX__
  packC8_common(weights, pack_weight, {0, 0, 0, 0}, 5, 5, ch_in);
#else
  packC4_common(weights, pack_weight, {0, 0, 0, 0}, 5, 5, ch_in);
#endif

  for (int n = 0; n < num; n++) {
    const float* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_out * size_out_channel;

    for (int c = 0; c < ch_out; c += block_channel) {
      int real_block_channel = Min(block_channel, ch_out - c);
      auto* dout_ptr = dout_batch + c * size_out_channel;
      auto* din_ptr = din_batch + c * size_in_channel;
      auto* weights_data = pack_weight + c * 5 * 5;

#ifdef __AVX__
      packC8_common(din_ptr,
                    pack_input,
                    {pad, pad, pad, pad},
                    h_in,
                    w_in,
                    real_block_channel);
#else
      packC4_common(din_ptr,
                    pack_input,
                    {pad, pad, pad, pad},
                    h_in,
                    w_in,
                    real_block_channel);
#endif

      float bias_ptr[block_channel] = {0.f};
      if (flag_bias) {
        for (int i = 0; i < block_channel; i++) {
          if (real_block_channel > i) {
            bias_ptr[i] = *(bias + c + i);
          }
        }
      }

      Type _bias = loadu_ps(bias_ptr);

      for (int i = 0; i < h_out; i++) {
        const float* block_inr0 = pack_input + i * 2 * in_len;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        int j = 0;
        float* dout_block = pack_out + i * w_out * block_channel;
        for (; j + 3 < w_out; j += 4) {
          Type i00 = loadu_ps(block_inr0);
          Type i01 = loadu_ps(block_inr0 + 1 * block_channel);
          Type i02 = loadu_ps(block_inr0 + 2 * block_channel);
          Type i03 = loadu_ps(block_inr0 + 3 * block_channel);
          Type i04 = loadu_ps(block_inr0 + 4 * block_channel);
          Type i05 = loadu_ps(block_inr0 + 5 * block_channel);
          Type i06 = loadu_ps(block_inr0 + 6 * block_channel);
          Type i07 = loadu_ps(block_inr0 + 7 * block_channel);
          Type i08 = loadu_ps(block_inr0 + 8 * block_channel);
          Type i09 = loadu_ps(block_inr0 + 9 * block_channel);
          Type i0a = loadu_ps(block_inr0 + 10 * block_channel);

          Type w00 = loadu_ps(weights_data);
          Type r0 = fmadd_ps(i00, w00, _bias);
          Type r1 = fmadd_ps(i02, w00, _bias);
          Type r2 = fmadd_ps(i04, w00, _bias);
          Type r3 = fmadd_ps(i06, w00, _bias);

          Type w01 = loadu_ps(weights_data + 1 * block_channel);
          r0 = fmadd_ps(i01, w01, r0);
          r1 = fmadd_ps(i03, w01, r1);
          r2 = fmadd_ps(i05, w01, r2);
          r3 = fmadd_ps(i07, w01, r3);

          Type w02 = loadu_ps(weights_data + 2 * block_channel);
          r0 = fmadd_ps(i02, w02, r0);
          r1 = fmadd_ps(i04, w02, r1);
          r2 = fmadd_ps(i06, w02, r2);
          r3 = fmadd_ps(i08, w02, r3);

          Type w03 = loadu_ps(weights_data + 3 * block_channel);
          r0 = fmadd_ps(i03, w03, r0);
          r1 = fmadd_ps(i05, w03, r1);
          r2 = fmadd_ps(i07, w03, r2);
          r3 = fmadd_ps(i09, w03, r3);

          Type w04 = loadu_ps(weights_data + 4 * block_channel);
          r0 = fmadd_ps(i04, w04, r0);
          r1 = fmadd_ps(i06, w04, r1);
          r2 = fmadd_ps(i08, w04, r2);
          r3 = fmadd_ps(i0a, w04, r3);

          Type i10 = loadu_ps(block_inr1);
          Type i11 = loadu_ps(block_inr1 + 1 * block_channel);
          Type i12 = loadu_ps(block_inr1 + 2 * block_channel);
          Type i13 = loadu_ps(block_inr1 + 3 * block_channel);
          Type i14 = loadu_ps(block_inr1 + 4 * block_channel);
          Type i15 = loadu_ps(block_inr1 + 5 * block_channel);
          Type i16 = loadu_ps(block_inr1 + 6 * block_channel);
          Type i17 = loadu_ps(block_inr1 + 7 * block_channel);
          Type i18 = loadu_ps(block_inr1 + 8 * block_channel);
          Type i19 = loadu_ps(block_inr1 + 9 * block_channel);
          Type i1a = loadu_ps(block_inr1 + 10 * block_channel);

          Type w10 = loadu_ps(weights_data + 5 * block_channel);
          r0 = fmadd_ps(i10, w10, r0);
          r1 = fmadd_ps(i12, w10, r1);
          r2 = fmadd_ps(i14, w10, r2);
          r3 = fmadd_ps(i16, w10, r3);

          Type w11 = loadu_ps(weights_data + 6 * block_channel);
          r0 = fmadd_ps(i11, w11, r0);
          r1 = fmadd_ps(i13, w11, r1);
          r2 = fmadd_ps(i15, w11, r2);
          r3 = fmadd_ps(i17, w11, r3);

          Type w12 = loadu_ps(weights_data + 7 * block_channel);
          r0 = fmadd_ps(i12, w12, r0);
          r1 = fmadd_ps(i14, w12, r1);
          r2 = fmadd_ps(i16, w12, r2);
          r3 = fmadd_ps(i18, w12, r3);

          Type w13 = loadu_ps(weights_data + 8 * block_channel);
          r0 = fmadd_ps(i13, w13, r0);
          r1 = fmadd_ps(i15, w13, r1);
          r2 = fmadd_ps(i17, w13, r2);
          r3 = fmadd_ps(i19, w13, r3);

          Type w14 = loadu_ps(weights_data + 9 * block_channel);
          r0 = fmadd_ps(i14, w14, r0);
          r1 = fmadd_ps(i16, w14, r1);
          r2 = fmadd_ps(i18, w14, r2);
          r3 = fmadd_ps(i1a, w14, r3);

          Type i20 = loadu_ps(block_inr2);
          Type i21 = loadu_ps(block_inr2 + 1 * block_channel);
          Type i22 = loadu_ps(block_inr2 + 2 * block_channel);
          Type i23 = loadu_ps(block_inr2 + 3 * block_channel);
          Type i24 = loadu_ps(block_inr2 + 4 * block_channel);
          Type i25 = loadu_ps(block_inr2 + 5 * block_channel);
          Type i26 = loadu_ps(block_inr2 + 6 * block_channel);
          Type i27 = loadu_ps(block_inr2 + 7 * block_channel);
          Type i28 = loadu_ps(block_inr2 + 8 * block_channel);
          Type i29 = loadu_ps(block_inr2 + 9 * block_channel);
          Type i2a = loadu_ps(block_inr2 + 10 * block_channel);

          Type w20 = loadu_ps(weights_data + 10 * block_channel);
          r0 = fmadd_ps(i20, w20, r0);
          r1 = fmadd_ps(i22, w20, r1);
          r2 = fmadd_ps(i24, w20, r2);
          r3 = fmadd_ps(i26, w20, r3);

          Type w21 = loadu_ps(weights_data + 11 * block_channel);
          r0 = fmadd_ps(i21, w21, r0);
          r1 = fmadd_ps(i23, w21, r1);
          r2 = fmadd_ps(i25, w21, r2);
          r3 = fmadd_ps(i27, w21, r3);

          Type w22 = loadu_ps(weights_data + 12 * block_channel);
          r0 = fmadd_ps(i22, w22, r0);
          r1 = fmadd_ps(i24, w22, r1);
          r2 = fmadd_ps(i26, w22, r2);
          r3 = fmadd_ps(i28, w22, r3);

          Type w23 = loadu_ps(weights_data + 13 * block_channel);
          r0 = fmadd_ps(i23, w23, r0);
          r1 = fmadd_ps(i25, w23, r1);
          r2 = fmadd_ps(i27, w23, r2);
          r3 = fmadd_ps(i29, w23, r3);

          Type w24 = loadu_ps(weights_data + 14 * block_channel);
          r0 = fmadd_ps(i24, w24, r0);
          r1 = fmadd_ps(i26, w24, r1);
          r2 = fmadd_ps(i28, w24, r2);
          r3 = fmadd_ps(i2a, w24, r3);

          Type i30 = loadu_ps(block_inr3);
          Type i31 = loadu_ps(block_inr3 + 1 * block_channel);
          Type i32 = loadu_ps(block_inr3 + 2 * block_channel);
          Type i33 = loadu_ps(block_inr3 + 3 * block_channel);
          Type i34 = loadu_ps(block_inr3 + 4 * block_channel);
          Type i35 = loadu_ps(block_inr3 + 5 * block_channel);
          Type i36 = loadu_ps(block_inr3 + 6 * block_channel);
          Type i37 = loadu_ps(block_inr3 + 7 * block_channel);
          Type i38 = loadu_ps(block_inr3 + 8 * block_channel);
          Type i39 = loadu_ps(block_inr3 + 9 * block_channel);
          Type i3a = loadu_ps(block_inr3 + 10 * block_channel);

          Type w30 = loadu_ps(weights_data + 15 * block_channel);
          r0 = fmadd_ps(i30, w30, r0);
          r1 = fmadd_ps(i32, w30, r1);
          r2 = fmadd_ps(i34, w30, r2);
          r3 = fmadd_ps(i36, w30, r3);

          Type w31 = loadu_ps(weights_data + 16 * block_channel);
          r0 = fmadd_ps(i31, w31, r0);
          r1 = fmadd_ps(i33, w31, r1);
          r2 = fmadd_ps(i35, w31, r2);
          r3 = fmadd_ps(i37, w31, r3);

          Type w32 = loadu_ps(weights_data + 17 * block_channel);
          r0 = fmadd_ps(i32, w32, r0);
          r1 = fmadd_ps(i34, w32, r1);
          r2 = fmadd_ps(i36, w32, r2);
          r3 = fmadd_ps(i38, w32, r3);

          Type w33 = loadu_ps(weights_data + 18 * block_channel);
          r0 = fmadd_ps(i33, w33, r0);
          r1 = fmadd_ps(i35, w33, r1);
          r2 = fmadd_ps(i37, w33, r2);
          r3 = fmadd_ps(i39, w33, r3);

          Type w34 = loadu_ps(weights_data + 19 * block_channel);
          r0 = fmadd_ps(i34, w34, r0);
          r1 = fmadd_ps(i36, w34, r1);
          r2 = fmadd_ps(i38, w34, r2);
          r3 = fmadd_ps(i3a, w34, r3);

          Type i40 = loadu_ps(block_inr4);
          Type i41 = loadu_ps(block_inr4 + 1 * block_channel);
          Type i42 = loadu_ps(block_inr4 + 2 * block_channel);
          Type i43 = loadu_ps(block_inr4 + 3 * block_channel);
          Type i44 = loadu_ps(block_inr4 + 4 * block_channel);
          Type i45 = loadu_ps(block_inr4 + 5 * block_channel);
          Type i46 = loadu_ps(block_inr4 + 6 * block_channel);
          Type i47 = loadu_ps(block_inr4 + 7 * block_channel);
          Type i48 = loadu_ps(block_inr4 + 8 * block_channel);
          Type i49 = loadu_ps(block_inr4 + 9 * block_channel);
          Type i4a = loadu_ps(block_inr4 + 10 * block_channel);

          Type w40 = loadu_ps(weights_data + 20 * block_channel);
          r0 = fmadd_ps(i40, w40, r0);
          r1 = fmadd_ps(i42, w40, r1);
          r2 = fmadd_ps(i44, w40, r2);
          r3 = fmadd_ps(i46, w40, r3);

          Type w41 = loadu_ps(weights_data + 21 * block_channel);
          r0 = fmadd_ps(i41, w41, r0);
          r1 = fmadd_ps(i43, w41, r1);
          r2 = fmadd_ps(i45, w41, r2);
          r3 = fmadd_ps(i47, w41, r3);

          Type w42 = loadu_ps(weights_data + 22 * block_channel);
          r0 = fmadd_ps(i42, w42, r0);
          r1 = fmadd_ps(i44, w42, r1);
          r2 = fmadd_ps(i46, w42, r2);
          r3 = fmadd_ps(i48, w42, r3);

          Type w43 = loadu_ps(weights_data + 23 * block_channel);
          r0 = fmadd_ps(i43, w43, r0);
          r1 = fmadd_ps(i45, w43, r1);
          r2 = fmadd_ps(i47, w43, r2);
          r3 = fmadd_ps(i49, w43, r3);

          Type w44 = loadu_ps(weights_data + 24 * block_channel);
          r0 = fmadd_ps(i44, w44, r0);
          r1 = fmadd_ps(i46, w44, r1);
          r2 = fmadd_ps(i48, w44, r2);
          r3 = fmadd_ps(i4a, w44, r3);

          Type zero = setzero_ps();
          if (has_active) {  // process activation
            if (act_type == lite_api::ActivationType::kRelu) {
              r0 = max_ps(r0, zero);
              r1 = max_ps(r1, zero);
              r2 = max_ps(r2, zero);
              r3 = max_ps(r3, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              Type six = set1_ps(act_param.Relu_clipped_coef);
              r0 = min_ps(max_ps(r0, zero), six);
              r1 = min_ps(max_ps(r1, zero), six);
              r2 = min_ps(max_ps(r2, zero), six);
              r3 = min_ps(max_ps(r3, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r0 = blendv_ps(
                  r0, mul_ps(negative_slope, r0), cmp_ps(r0, zero, 2));
              r1 = blendv_ps(
                  r1, mul_ps(negative_slope, r1), cmp_ps(r1, zero, 2));
              r2 = blendv_ps(
                  r2, mul_ps(negative_slope, r2), cmp_ps(r2, zero, 2));
              r3 = blendv_ps(
                  r3, mul_ps(negative_slope, r3), cmp_ps(r3, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              Type vscale = set1_ps(1.0 / act_param.hard_swish_scale);
              Type voffset = set1_ps(act_param.hard_swish_offset);
              Type vthreshold = set1_ps(act_param.hard_swish_threshold);
              r0 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r0, voffset))),
                          mul_ps(r0, vscale));
              r1 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r1, voffset))),
                          mul_ps(r1, vscale));
              r2 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r2, voffset))),
                          mul_ps(r2, vscale));
              r3 = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r3, voffset))),
                          mul_ps(r3, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << " not supported ";
            }
          }

          storeu_ps(dout_block, r0);
          storeu_ps(dout_block + 1 * block_channel, r1);
          storeu_ps(dout_block + 2 * block_channel, r2);
          storeu_ps(dout_block + 3 * block_channel, r3);
          dout_block += 4 * block_channel;

          block_inr0 += 8 * block_channel;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        for (; j < w_out; j++) {
          Type r = _bias;
          for (int m = 0; m < 5; m++) {
            for (int n = 0; n < 5; n++) {
              Type weight = loadu_ps(weights_data + 5 * block_channel * m +
                                     block_channel * n);
              Type input = loadu_ps(block_inr0 + block_channel * (j % 4) * 2 +
                                    in_len * m + block_channel * n);
              r = fmadd_ps(input, weight, r);
            }
          }
          Type zero = setzero_ps();
          if (has_active) {  // process activation
            if (act_type == lite_api::ActivationType::kRelu) {
              r = max_ps(r, zero);
            } else if (act_type == lite_api::ActivationType::kRelu6) {
              Type six = set1_ps(act_param.Relu_clipped_coef);
              r = min_ps(max_ps(r, zero), six);
            } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
              Type negative_slope = set1_ps(act_param.Leaky_relu_alpha);
              r = blendv_ps(r, mul_ps(negative_slope, r), cmp_ps(r, zero, 2));
            } else if (act_type == lite_api::ActivationType::kHardSwish) {
              Type vscale = set1_ps(1.0 / act_param.hard_swish_scale);
              Type voffset = set1_ps(act_param.hard_swish_offset);
              Type vthreshold = set1_ps(act_param.hard_swish_threshold);
              r = mul_ps(min_ps(vthreshold, max_ps(zero, add_ps(r, voffset))),
                         mul_ps(r, vscale));
            } else {
              LOG(FATAL) << " [X86] activation type "
                         << static_cast<int>(act_type) << "not supported";
            }
          }
          storeu_ps(dout_block, r);
          dout_block += block_channel;
        }
      }

#ifdef __AVX__
      unpackC8_common(pack_out, dout_ptr, size_out_channel, real_block_channel);
#else
      unpackC4_common(pack_out, dout_ptr, size_out_channel, real_block_channel);
#endif
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
