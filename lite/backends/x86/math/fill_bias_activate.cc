/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/fill_bias_activate.h"
#include <string.h>
#include <algorithm>
#include "lite/core/op_registry.h"

#ifdef __AVX__
#include <immintrin.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

static void activate_relu_inplace(float *data, int len, float alpha, int mode) {
  int i = 0;

  if (0 == mode) {  // relu
#ifdef __AVX__
    __m256 vec_zero = _mm256_set1_ps(0.f);
    for (; i + 7 < len; i += 8) {
      __m256 vec_data = _mm256_loadu_ps(data + i);
      _mm256_storeu_ps(data + i, _mm256_max_ps(vec_data, vec_zero));
    }
#endif
#ifdef __SSE__
    __m128 vec_zero_128 = _mm_set1_ps(0.f);
    for (; i + 3 < len; i += 4) {
      __m128 vec_data_128 = _mm_loadu_ps(data + i);
      _mm_storeu_ps(data + i, _mm_max_ps(vec_data_128, vec_zero_128));
    }
#endif
    for (; i < len; i++) {
      data[i] = data[i] > 0.f ? data[i] : 0.f;
    }
  } else {  // relu6
#ifdef __AVX__
    __m256 vec_zero = _mm256_set1_ps(0.f);
    __m256 vec_alph = _mm256_set1_ps(alpha);
    for (; i + 7 < len; i += 8) {
      __m256 vec_data = _mm256_loadu_ps(data + i);
      _mm256_storeu_ps(
          data + i, _mm256_min_ps(_mm256_max_ps(vec_data, vec_zero), vec_alph));
    }
#endif
#ifdef __SSE__
    __m128 vec_zero_128 = _mm_set1_ps(0.f);
    __m128 vec_alph_128 = _mm_set1_ps(alpha);
    for (; i + 3 < len; i += 4) {
      __m128 vec_data_128 = _mm_loadu_ps(data + i);
      _mm_storeu_ps(
          data + i,
          _mm_min_ps(_mm_max_ps(vec_data_128, vec_zero_128), vec_alph_128));
    }
#endif
    for (; i < len; i++) {
      data[i] = data[i] > 0.f ? data[i] : 0.f;
      data[i] = data[i] < alpha ? data[i] : alpha;
    }
  }
}

static void activate_relu_inplace_bias(float *data,
                                       const float *bias,
                                       int channel,
                                       int channel_size,
                                       float alpha,
                                       int mode) {
  int i = 0;
  int j = 0;
  float *tmp_data = data;

#ifdef __AVX__
  __m256 vec_zero = {0.f};
  __m256 vec_bias = {0.f};
  __m256 vec_data = {0.f};
  __m256 vec_alph = _mm256_set1_ps(alpha);
#endif
#ifdef __SSE__
  __m128 vec_zero_128 = {0.f};
  __m128 vec_bias_128 = {0.f};
  __m128 vec_data_128 = {0.f};
  __m128 vec_alph_128 = _mm_set1_ps(alpha);
#endif

  if (0 == mode) {  // relu
    for (j = 0; j < channel; j++) {
      i = 0;
      tmp_data = data + j * channel_size;
#ifdef __AVX__
      vec_bias = _mm256_set1_ps(bias[j]);
      for (; i + 7 < channel_size; i += 8) {
        vec_data = _mm256_loadu_ps(tmp_data + i);
        vec_data = _mm256_add_ps(vec_bias, vec_data);
        _mm256_storeu_ps(tmp_data + i, _mm256_max_ps(vec_data, vec_zero));
      }
#endif
#ifdef __SSE__
      vec_bias_128 = _mm_set1_ps(bias[j]);
      for (; i + 3 < channel_size; i += 4) {
        vec_data_128 = _mm_loadu_ps(tmp_data + i);
        vec_data_128 = _mm_add_ps(vec_data_128, vec_bias_128);
        _mm_storeu_ps(tmp_data + i, _mm_max_ps(vec_data_128, vec_zero_128));
      }
#endif
      for (; i < channel_size; i++) {
        tmp_data[i] += bias[j];
        tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
      }
    }
  } else {  // relu6
    for (j = 0; j < channel; j++) {
      i = 0;
      tmp_data = data + j * channel_size;
#ifdef __AVX__
      vec_bias = _mm256_set1_ps(bias[j]);
      for (; i + 7 < channel_size; i += 8) {
        vec_data = _mm256_loadu_ps(tmp_data + i);
        vec_data = _mm256_add_ps(vec_bias, vec_data);
        _mm256_storeu_ps(
            tmp_data + i,
            _mm256_min_ps(_mm256_max_ps(vec_data, vec_zero), vec_alph));
      }
#endif
#ifdef __SSE__
      vec_bias_128 = _mm_set1_ps(bias[j]);
      for (; i + 3 < channel_size; i += 4) {
        vec_data_128 = _mm_loadu_ps(tmp_data + i);
        vec_data_128 = _mm_add_ps(vec_data_128, vec_bias_128);
        _mm_storeu_ps(
            tmp_data + i,
            _mm_min_ps(_mm_max_ps(vec_data_128, vec_zero_128), vec_alph_128));
      }
#endif
      for (; i < channel_size; i++) {
        tmp_data[i] += bias[j];
        tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
        tmp_data[i] = tmp_data[i] < alpha ? tmp_data[i] : alpha;
      }
    }
  }
}

static void activate_lrelu_inplace(float *data, int len, float alpha) {
  const int cmp_le_os = 2;
  int i = 0;

#ifdef __AVX__
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_alph = _mm256_set1_ps(alpha);
  for (; i + 7 < len; i += 8) {
    __m256 vec_data = _mm256_loadu_ps(data + i);
    __m256 vec_lr = _mm256_mul_ps(vec_alph, vec_data);
    __m256 vec_mask = _mm256_cmp_ps(vec_data, vec_zero, cmp_le_os);
    _mm256_storeu_ps(data + i, _mm256_blendv_ps(vec_data, vec_lr, vec_mask));
  }
#endif
#ifdef __SSE4_1__  // blendv need 4.1
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_alph_128 = _mm_set1_ps(alpha);
  for (; i + 3 < len; i += 4) {
    __m128 vec_data_128 = _mm_loadu_ps(data + i);
    __m128 vec_lr_128 = _mm_mul_ps(vec_data_128, vec_alph_128);
    __m128 vec_mask_128 = _mm_cmple_ps(vec_data_128, vec_zero_128);
    _mm_storeu_ps(data + i,
                  _mm_blendv_ps(vec_data_128, vec_lr_128, vec_mask_128));
  }
#endif
  for (; i < len; i++) {
    data[i] = data[i] > 0.f ? data[i] : alpha * data[i];
  }
}

static void activate_lrelu_inplace_bias(float *data,
                                        const float *bias,
                                        int channel,
                                        int channel_size,
                                        float alpha) {
  const int cmp_le_os = 2;
  int i = 0;
  int j = 0;
  float *tmp_data = data;

#ifdef __AVX__
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_alph = _mm256_set1_ps(alpha);
  __m256 vec_bias = {0.f};
#endif
#ifdef __SSE4_1__
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_alph_128 = _mm_set1_ps(alpha);
  __m128 vec_bias_128 = {0.f};
#endif

  for (j = 0; j < channel; j++) {
    i = 0;
    tmp_data = data + j * channel_size;

#ifdef __AVX__
    vec_bias = _mm256_set1_ps(bias[j]);
    for (; i + 7 < channel_size; i += 8) {
      __m256 vec_data = _mm256_add_ps(vec_bias, _mm256_loadu_ps(tmp_data + i));
      __m256 vec_lr = _mm256_mul_ps(vec_alph, vec_data);
      __m256 vec_mask = _mm256_cmp_ps(vec_data, vec_zero, cmp_le_os);
      _mm256_storeu_ps(tmp_data + i,
                       _mm256_blendv_ps(vec_data, vec_lr, vec_mask));
    }
#endif
#ifdef __SSE4_1__
    vec_bias_128 = _mm_set1_ps(bias[j]);
    for (; i + 3 < channel_size; i += 4) {
      __m128 vec_data_128 =
          _mm_add_ps(vec_bias_128, _mm_loadu_ps(tmp_data + i));
      __m128 vec_lr_128 = _mm_mul_ps(vec_data_128, vec_alph_128);
      __m128 vec_mask_128 = _mm_cmple_ps(vec_data_128, vec_zero_128);
      _mm_storeu_ps(tmp_data + i,
                    _mm_blendv_ps(vec_data_128, vec_lr_128, vec_mask_128));
    }
#endif
    for (; i < channel_size; i++) {
      tmp_data[i] += bias[j];
      tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : alpha * tmp_data[i];
    }
  }
}

static void activate_hardswish_inplace_bias(float *data,
                                            const float *bias,
                                            int channel,
                                            int channel_size,
                                            float scale,
                                            float threshold,
                                            float offset) {
#ifdef __AVX__
  int cnt = channel_size >> 5;
  int remain = channel_size & 31;
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_scale = _mm256_set1_ps(1.0 / scale);
  __m256 vec_threshold = _mm256_set1_ps(threshold);
  __m256 vec_offset = _mm256_set1_ps(offset);
#else
  int cnt = channel_size >> 4;
  int remain = channel_size & 15;
#endif
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_scale_128 = _mm_set1_ps(1.0 / scale);
  __m128 vec_threshold_128 = _mm_set1_ps(threshold);
  __m128 vec_offset_128 = _mm_set1_ps(offset);
  int cnt_4 = remain >> 2;
  int rem_4 = remain & 3;
  for (int i = 0; i < channel; i++) {
#ifdef __AVX__
    __m256 vec_bias = _mm256_set1_ps(bias[i]);
#endif
    __m128 vec_bias_128 = _mm_set1_ps(bias[i]);
    float *tmp_data = data + i * channel_size;

    for (int j = 0; j < cnt; j++) {
#ifdef __AVX__
      __m256 vin0 = _mm256_add_ps(_mm256_loadu_ps(tmp_data), vec_bias);
      __m256 vin1 = _mm256_add_ps(_mm256_loadu_ps(tmp_data + 8), vec_bias);
      __m256 vin2 = _mm256_add_ps(_mm256_loadu_ps(tmp_data + 16), vec_bias);
      __m256 vin3 = _mm256_add_ps(_mm256_loadu_ps(tmp_data + 24), vec_bias);
      __m256 vadd0 = _mm256_add_ps(vin0, vec_offset);
      __m256 vadd1 = _mm256_add_ps(vin1, vec_offset);
      __m256 vadd2 = _mm256_add_ps(vin2, vec_offset);
      __m256 vadd3 = _mm256_add_ps(vin3, vec_offset);
      __m256 vsum0 = _mm256_mul_ps(vin0, vec_scale);
      __m256 vsum1 = _mm256_mul_ps(vin1, vec_scale);
      __m256 vsum2 = _mm256_mul_ps(vin2, vec_scale);
      __m256 vsum3 = _mm256_mul_ps(vin3, vec_scale);
      __m256 vres0 =
          _mm256_min_ps(_mm256_max_ps(vadd0, vec_zero), vec_threshold);
      __m256 vres1 =
          _mm256_min_ps(_mm256_max_ps(vadd1, vec_zero), vec_threshold);
      __m256 vres2 =
          _mm256_min_ps(_mm256_max_ps(vadd2, vec_zero), vec_threshold);
      __m256 vres3 =
          _mm256_min_ps(_mm256_max_ps(vadd3, vec_zero), vec_threshold);
      _mm256_storeu_ps(tmp_data, _mm256_mul_ps(vres0, vsum0));
      _mm256_storeu_ps(tmp_data + 8, _mm256_mul_ps(vres1, vsum1));
      _mm256_storeu_ps(tmp_data + 16, _mm256_mul_ps(vres2, vsum2));
      _mm256_storeu_ps(tmp_data + 24, _mm256_mul_ps(vres3, vsum3));
      tmp_data += 32;
#else
      __m128 vin0 = _mm_add_ps(_mm_loadu_ps(tmp_data), vec_bias_128);
      __m128 vin1 = _mm_add_ps(_mm_loadu_ps(tmp_data + 4), vec_bias_128);
      __m128 vin2 = _mm_add_ps(_mm_loadu_ps(tmp_data + 8), vec_bias_128);
      __m128 vin3 = _mm_add_ps(_mm_loadu_ps(tmp_data + 12), vec_bias_128);
      __m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
      __m128 vadd1 = _mm_add_ps(vin1, vec_offset_128);
      __m128 vadd2 = _mm_add_ps(vin2, vec_offset_128);
      __m128 vadd3 = _mm_add_ps(vin3, vec_offset_128);
      __m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
      __m128 vsum1 = _mm_mul_ps(vin1, vec_scale_128);
      __m128 vsum2 = _mm_mul_ps(vin2, vec_scale_128);
      __m128 vsum3 = _mm_mul_ps(vin3, vec_scale_128);
      __m128 vres0 =
          _mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
      __m128 vres1 =
          _mm_min_ps(_mm_max_ps(vadd1, vec_zero_128), vec_threshold_128);
      __m128 vres2 =
          _mm_min_ps(_mm_max_ps(vadd2, vec_zero_128), vec_threshold_128);
      __m128 vres3 =
          _mm_min_ps(_mm_max_ps(vadd3, vec_zero_128), vec_threshold_128);
      _mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
      _mm_storeu_ps(tmp_data + 4, _mm_mul_ps(vres1, vsum1));
      _mm_storeu_ps(tmp_data + 8, _mm_mul_ps(vres2, vsum2));
      _mm_storeu_ps(tmp_data + 12, _mm_mul_ps(vres3, vsum3));
      tmp_data += 16;
#endif
    }
    for (int j = 0; j < cnt_4; j++) {
      __m128 vin0 = _mm_add_ps(_mm_loadu_ps(tmp_data), vec_bias_128);
      __m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
      __m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
      __m128 vres0 =
          _mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
      _mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
      tmp_data += 4;
    }
    for (int j = 0; j < rem_4; j++) {
      tmp_data[0] = tmp_data[0] + bias[i];
      tmp_data[0] = std::min(std::max(0.f, tmp_data[0] + offset), threshold) *
                    tmp_data[0] / scale;
      tmp_data++;
    }
  }
}

static void activate_hardswish_inplace(
    float *data, int len, float scale, float threshold, float offset) {
#ifdef __AVX__
  int cnt = len >> 5;
  int remain = len & 31;
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_scale = _mm256_set1_ps(1.0 / scale);
  __m256 vec_threshold = _mm256_set1_ps(threshold);
  __m256 vec_offset = _mm256_set1_ps(offset);
#else
  int cnt = len >> 4;
  int remain = len & 15;
#endif
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_scale_128 = _mm_set1_ps(1.0 / scale);
  __m128 vec_threshold_128 = _mm_set1_ps(threshold);
  __m128 vec_offset_128 = _mm_set1_ps(offset);
  int cnt_4 = remain >> 2;
  int rem_4 = remain & 3;
  float *tmp_data = data;
  for (int i = 0; i < cnt; i++) {
#ifdef __AVX__
    __m256 vin0 = _mm256_loadu_ps(tmp_data);
    __m256 vin1 = _mm256_loadu_ps(tmp_data + 8);
    __m256 vin2 = _mm256_loadu_ps(tmp_data + 16);
    __m256 vin3 = _mm256_loadu_ps(tmp_data + 24);
    __m256 vadd0 = _mm256_add_ps(vin0, vec_offset);
    __m256 vadd1 = _mm256_add_ps(vin1, vec_offset);
    __m256 vadd2 = _mm256_add_ps(vin2, vec_offset);
    __m256 vadd3 = _mm256_add_ps(vin3, vec_offset);
    __m256 vsum0 = _mm256_mul_ps(vin0, vec_scale);
    __m256 vsum1 = _mm256_mul_ps(vin1, vec_scale);
    __m256 vsum2 = _mm256_mul_ps(vin2, vec_scale);
    __m256 vsum3 = _mm256_mul_ps(vin3, vec_scale);
    __m256 vres0 = _mm256_min_ps(_mm256_max_ps(vadd0, vec_zero), vec_threshold);
    __m256 vres1 = _mm256_min_ps(_mm256_max_ps(vadd1, vec_zero), vec_threshold);
    __m256 vres2 = _mm256_min_ps(_mm256_max_ps(vadd2, vec_zero), vec_threshold);
    __m256 vres3 = _mm256_min_ps(_mm256_max_ps(vadd3, vec_zero), vec_threshold);
    _mm256_storeu_ps(tmp_data, _mm256_mul_ps(vres0, vsum0));
    _mm256_storeu_ps(tmp_data + 8, _mm256_mul_ps(vres1, vsum1));
    _mm256_storeu_ps(tmp_data + 16, _mm256_mul_ps(vres2, vsum2));
    _mm256_storeu_ps(tmp_data + 24, _mm256_mul_ps(vres3, vsum3));
    tmp_data += 32;
#else
    __m128 vin0 = _mm_loadu_ps(tmp_data);
    __m128 vin1 = _mm_loadu_ps(tmp_data + 4);
    __m128 vin2 = _mm_loadu_ps(tmp_data + 8);
    __m128 vin3 = _mm_loadu_ps(tmp_data + 12);
    __m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
    __m128 vadd1 = _mm_add_ps(vin1, vec_offset_128);
    __m128 vadd2 = _mm_add_ps(vin2, vec_offset_128);
    __m128 vadd3 = _mm_add_ps(vin3, vec_offset_128);
    __m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
    __m128 vsum1 = _mm_mul_ps(vin1, vec_scale_128);
    __m128 vsum2 = _mm_mul_ps(vin2, vec_scale_128);
    __m128 vsum3 = _mm_mul_ps(vin3, vec_scale_128);
    __m128 vres0 =
        _mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
    __m128 vres1 =
        _mm_min_ps(_mm_max_ps(vadd1, vec_zero_128), vec_threshold_128);
    __m128 vres2 =
        _mm_min_ps(_mm_max_ps(vadd2, vec_zero_128), vec_threshold_128);
    __m128 vres3 =
        _mm_min_ps(_mm_max_ps(vadd3, vec_zero_128), vec_threshold_128);
    _mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
    _mm_storeu_ps(tmp_data + 4, _mm_mul_ps(vres1, vsum1));
    _mm_storeu_ps(tmp_data + 8, _mm_mul_ps(vres2, vsum2));
    _mm_storeu_ps(tmp_data + 12, _mm_mul_ps(vres3, vsum3));
    tmp_data += 16;
#endif
  }
  for (int i = 0; i < cnt_4; i++) {
    __m128 vin0 = _mm_loadu_ps(tmp_data);
    __m128 vadd0 = _mm_add_ps(vin0, vec_offset_128);
    __m128 vsum0 = _mm_mul_ps(vin0, vec_scale_128);
    __m128 vres0 =
        _mm_min_ps(_mm_max_ps(vadd0, vec_zero_128), vec_threshold_128);
    _mm_storeu_ps(tmp_data, _mm_mul_ps(vres0, vsum0));
    tmp_data += 4;
  }
  for (int i = 0; i < rem_4; i++) {
    tmp_data[0] = std::min(std::max(0.f, tmp_data[0] + offset), threshold) *
                  tmp_data[0] / scale;
    tmp_data++;
  }
}

static void activate_none_inplace_bias(float *data,
                                       const float *bias,
                                       int channel,
                                       int channel_size) {
  int i = 0;
  int j = 0;
  float *tmp_data = data;

#ifdef __AVX__
  __m256 vec_bias = {0.f};
  __m256 vec_data = {0.f};
#endif
#ifdef __SSE__
  __m128 vec_bias_128 = {0.f};
  __m128 vec_data_128 = {0.f};
#endif

  for (j = 0; j < channel; j++) {
    i = 0;
    tmp_data = data + j * channel_size;
#ifdef __AVX__
    vec_bias = _mm256_set1_ps(bias[j]);
    for (; i + 7 < channel_size; i += 8) {
      vec_data = _mm256_loadu_ps(tmp_data + i);
      vec_data = _mm256_add_ps(vec_bias, vec_data);
      _mm256_storeu_ps(tmp_data + i, vec_data);
    }
#endif
#ifdef __SSE__
    vec_bias_128 = _mm_set1_ps(bias[j]);
    for (; i + 3 < channel_size; i += 4) {
      vec_data_128 = _mm_loadu_ps(tmp_data + i);
      vec_data_128 = _mm_add_ps(vec_data_128, vec_bias_128);
      _mm_storeu_ps(tmp_data + i, vec_data_128);
    }
#endif
    for (; i < channel_size; i++) {
      tmp_data[i] += bias[j];
    }
  }
}

void fill_bias_act(float *tensor,
                   const float *bias,
                   int channel,
                   int channel_size,
                   bool flag_bias,
                   const operators::ActivationParam *act_param) {
  auto act_type = act_param->active_type;
  float local_alpha = 0.f;
  int len = channel * channel_size;

  if ((act_param != nullptr) && (act_param->has_active)) {
    if ((flag_bias) && (bias != nullptr)) {
      // activate and bias
      if (act_type == lite_api::ActivationType::kRelu) {
        activate_relu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha, 0);
      } else if (act_type == lite_api::ActivationType::kRelu6) {
        local_alpha = act_param->Relu_clipped_coef;
        activate_relu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha, 1);
      } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
        local_alpha = act_param->Leaky_relu_alpha;
        activate_lrelu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha);
      } else if (act_type == lite_api::ActivationType::kHardSwish) {
        local_alpha = act_param->hard_swish_scale;
        activate_hardswish_inplace_bias(tensor,
                                        bias,
                                        channel,
                                        channel_size,
                                        local_alpha,
                                        act_param->hard_swish_threshold,
                                        act_param->hard_swish_offset);
      }
    } else {
      // activate
      if (act_type == lite_api::ActivationType::kRelu) {
        activate_relu_inplace(tensor, len, local_alpha, 0);
      } else if (act_type == lite_api::ActivationType::kRelu6) {
        local_alpha = act_param->Relu_clipped_coef;
        activate_relu_inplace(tensor, len, local_alpha, 1);
      } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
        local_alpha = act_param->Leaky_relu_alpha;
        activate_lrelu_inplace(tensor, len, local_alpha);
      } else if (act_type == lite_api::ActivationType::kHardSwish) {
        local_alpha = act_param->hard_swish_scale;
        activate_hardswish_inplace(tensor,
                                   len,
                                   local_alpha,
                                   act_param->hard_swish_threshold,
                                   act_param->hard_swish_offset);
      }
    }
  } else {
    // only add bias
    if ((flag_bias) && (bias != nullptr))
      activate_none_inplace_bias(tensor, bias, channel, channel_size);
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
