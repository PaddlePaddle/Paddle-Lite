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

#ifdef __AVX__
#include <immintrin.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include "lite/backends/x86/math/activation_functions.h"
#include "lite/backends/x86/math/rnn.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

using namespace paddle::lite::x86::math::detail::forward;

template <typename T>
void act_relu(const T* din, T* dout, int size, int threads);

template <typename T>
void act_sigmoid(const T* din, T* dout, int size, int threads);

template <typename T>
void act_tanh(const T* din, T* dout, int size, int threads);

void vector_dot(
    float* out, const float* in, const float* v1, int size, const float* v2) {
#if defined(__AVX__)
  __m256 vec_in, vec_v1, vec_v2;
#endif
#if defined(__SSE__)
  __m128 vec_in_128, vec_v1_128, vec_v2_128;
#endif

  int i = 0;
  if (nullptr == v2) {
    i = 0;

// in_out * v1
#if defined(__AVX__)
    for (; i + 7 < size; i += 8) {
      vec_in = _mm256_loadu_ps(in + i);
      vec_v1 = _mm256_loadu_ps(v1 + i);
      _mm256_storeu_ps(out + i, _mm256_mul_ps(vec_in, vec_v1));
    }
    _mm256_zeroupper();
#endif
#if defined(__SSE__)
    for (; i + 3 < size; i += 4) {
      vec_in_128 = _mm_loadu_ps(in + i);
      vec_v1_128 = _mm_loadu_ps(v1 + i);
      _mm_storeu_ps(out + i, _mm_mul_ps(vec_in_128, vec_v1_128));
    }
#endif
    for (; i < size; i++) {
      out[i] = in[i] * v1[i];
    }
  } else {
    i = 0;

// in_out + v1 * v2
#if defined(__AVX__) && defined(__FMA__)
    for (; i + 7 < size; i += 8) {
      vec_in = _mm256_loadu_ps(in + i);
      vec_v1 = _mm256_loadu_ps(v1 + i);
      vec_v2 = _mm256_loadu_ps(v2 + i);
      _mm256_storeu_ps(out + i, _mm256_fmadd_ps(vec_v2, vec_v1, vec_in));
    }
    for (; i + 3 < size; i += 4) {
      vec_in_128 = _mm_loadu_ps(in + i);
      vec_v1_128 = _mm_loadu_ps(v1 + i);
      vec_v2_128 = _mm_loadu_ps(v2 + i);
      _mm_storeu_ps(out + i, _mm_fmadd_ps(vec_v2_128, vec_v1_128, vec_in_128));
    }
#endif
    for (; i < size; i++) {
      out[i] = in[i] + v1[i] * v2[i];
    }
  }
}

template <>
void act_relu<float>(const float* din, float* dout, int size, int threads) {
  int i = 0;

#ifdef __AVX__
  for (; i + 7 < size; i += 8) {
    __m256 a = _mm256_loadu_ps(din + i);
    _mm256_storeu_ps(dout + i, avx::Relu(a));
  }
#endif
  for (; i < size; i++) {
    dout[i] = Relu<float>(din[i]);
  }
}

template <>
void act_sigmoid<float>(const float* din, float* dout, int size, int threads) {
  int i = 0;

#ifdef __AVX__
  for (; i + 7 < size; i += 8) {
    __m256 a = _mm256_loadu_ps(din + i);
    _mm256_storeu_ps(dout + i, avx::Sigmoid(a));
  }
#endif
  for (; i < size; i++) {
    dout[i] = Sigmoid<float>(din[i]);
  }
}

template <>
void act_tanh<float>(const float* din, float* dout, int size, int threads) {
  int i = 0;

#ifdef __AVX__
  for (; i + 7 < size; i += 8) {
    __m256 a = _mm256_loadu_ps(din + i);
    _mm256_storeu_ps(dout + i, avx::Tanh(a));
  }
#endif
  for (; i < size; i++) {
    dout[i] = Tanh<float>(din[i]);
  }
}

void fill_bias_fc(float* out, const float* bias, int num, int channel) {
#ifdef __AVX__
  __m256 vec_bias = {0.f};
  __m256 vec_data = {0.f};
#endif
#ifdef __SSE__
  __m128 vec_bias_128 = {0.f};
  __m128 vec_data_128 = {0.f};
#endif
  int i = 0;

  for (int j = 0; j < num; j++) {
    float* ptr = out + j * channel;
    const float* pbias = bias;
    i = 0;

#ifdef __AVX__
    for (; i + 7 < channel; i += 8) {
      vec_bias = _mm256_loadu_ps(pbias + i);
      vec_data = _mm256_loadu_ps(ptr + i);
      _mm256_storeu_ps(ptr + i, _mm256_add_ps(vec_data, vec_bias));
    }
#endif
#ifdef __SSE__
    for (; i + 3 < channel; i += 4) {
      vec_bias_128 = _mm_loadu_ps(pbias + i);
      vec_data_128 = _mm_loadu_ps(ptr + i);
      _mm_storeu_ps(ptr + i, _mm_add_ps(vec_data_128, vec_bias_128));
    }
#endif
    for (; i < channel; i++) {
      *(ptr + i) = pbias[i] + ptr[i];
    }
  }
}

template <typename T>
void activation(
    const T* din, T* dout, int size, std::string act_str, int threads) {
  if (act_str == "sigmoid") {
    act_sigmoid(din, dout, size, threads);
  } else if (act_str == "tanh") {
    act_tanh(din, dout, size, threads);
  } else if (act_str == "relu") {
    act_relu(din, dout, size, threads);
  } else {
    LOG(FATAL) << "unsupport activation " << act_str;
  }
}

template <typename T>
void activation(const T* din,
                T* dout,
                int size,
                lite_api::ActivationType act_type,
                int threads) {
  switch (act_type) {
    case lite_api::ActivationType::kSigmoid:
      act_sigmoid(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kSigmoid_v2:
      act_sigmoid(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kTanh:
      act_tanh(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kTanh_v2:
      act_tanh(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kRelu:
      act_relu(din, dout, size, threads);
      break;
    default:
      LOG(FATAL) << "unsupport activation type:" << static_cast<int>(act_type);
      break;
  }
}


template <typename T>
void GruRnnComputeKernel(GRUMetaValue<T> value,
                         int frame_size,
                         int batch_size,
                         lite_api::ActivationType active_node,
                         lite_api::ActivationType active_gate) {
  auto value_reset_gate = value.gate_value;
  auto value_update_gate = value.gate_value + frame_size;
  auto value_reset_output = value.reset_output_value;
  auto value_reset_bias = value.reset_bias;
  auto cell_state_value = value.gate_value + 2 * frame_size;
  auto value_output = value.output_value;
  auto value_prev_out = value.prev_out_value;

  for (int b = 0; b < batch_size; b++) {
    activation(value_reset_gate,
               value_reset_gate,
               frame_size,
               lite_api::ActivationType::kSigmoid_v2,
               1);

    activation(value_update_gate,
               value_update_gate,
               frame_size,
               lite_api::ActivationType::kSigmoid_v2,
               1);

    for (int i = 0; i < frame_size; i++) {
      value_reset_output[i] =
          (value_reset_output[i] + value_reset_bias[i]) * value_reset_gate[i];
      cell_state_value[i] += value_reset_output[i];
    }

    activation(cell_state_value,
               cell_state_value,
               frame_size,
               lite_api::ActivationType::kTanh_v2,
               1);

    if (value.prev_out_value) {
      for (int i = 0; i < frame_size; i++) {
        value_output[i] = (1.f - value_update_gate[i]) * cell_state_value[i] +
                           value_update_gate[i] * value_prev_out[i];
      }
    } else {
      for (int i = 0; i < frame_size; i++) {
        value_output[i] = (1.f - value_update_gate[i]) * cell_state_value[i];
      }
    }

    value_reset_gate += frame_size * 3;
    value_update_gate += frame_size * 3;
    value_reset_output += frame_size;
    cell_state_value += frame_size * 3;
    value_output += frame_size;
    if (value.prev_out_value) {
      value_prev_out += frame_size;
    }
  }
}

template <>
void GruRnnComputeKernel<float>(GRUMetaValue<float> value,
                                int frame_size,
                                int batch_size,
                                lite_api::ActivationType active_node,
                                lite_api::ActivationType active_gate) {
  auto value_reset_gate = value.gate_value;
  auto value_update_gate = value.gate_value + frame_size;
  auto value_reset_output = value.reset_output_value;
  auto value_reset_bias = value.reset_bias;
  auto cell_state_value = value.gate_value + 2 * frame_size;
  auto value_output = value.output_value;
  auto value_prev_out = value.prev_out_value;
  int i = 0;

#ifdef __AVX__
  __m256 vec_one_256 = _mm256_set1_ps(1.0f);
#endif
#ifdef __SSE__
  __m128 vec_one_128 = _mm_set1_ps(1.0f);
#endif

  for (int b = 0; b < batch_size; b++) {
    activation(value_reset_gate,
               value_reset_gate,
               frame_size,
               lite_api::ActivationType::kSigmoid_v2,
               1);
    activation(value_update_gate,
               value_update_gate,
               frame_size,
               lite_api::ActivationType::kSigmoid_v2,
               1);
    i = 0;
  #ifdef __AVX__
    for (; i + 7 < frame_size; i += 8) {
      __m256 vec_out = _mm256_loadu_ps(value_reset_output + i);
      __m256 vec_reset = _mm256_loadu_ps(value_reset_gate + i);
      __m256 vec_bias = _mm256_loadu_ps(value_reset_bias + i);
      vec_out = _mm256_mul_ps(_mm256_add_ps(vec_out, vec_bias), vec_reset);
      _mm256_storeu_ps(value_reset_output + i, vec_out);
      _mm256_storeu_ps(cell_state_value + i, 
     	 _mm256_add_ps(vec_out, _mm256_loadu_ps(cell_state_value + i)));
    }
  #endif
  #ifdef __SSE__
    for (; i + 3 < frame_size; i += 4) {
      __m128 vec_out = _mm_loadu_ps(value_reset_output + i);
      __m128 vec_reset = _mm_loadu_ps(value_reset_gate + i);
      __m128 vec_bias = _mm_loadu_ps(value_reset_bias + i);
      vec_out = _mm_mul_ps(_mm_add_ps(vec_out, vec_bias), vec_reset);
      _mm_storeu_ps(value_reset_output + i, vec_out);
      _mm_storeu_ps(cell_state_value + i, 
     	 _mm_add_ps(vec_out, _mm_loadu_ps(cell_state_value + i)));
    }
  #endif
    for (; i < frame_size; i++) {
      value_reset_output[i] =
          (value_reset_output[i] + value_reset_bias[i]) * value_reset_gate[i];
      cell_state_value[i] += value_reset_output[i];
    }

    activation(cell_state_value,
               cell_state_value,
               frame_size,
               lite_api::ActivationType::kTanh_v2,
               1);

    if (value.prev_out_value) {
      i = 0;
#ifdef __AVX__
      for (; i + 7 < frame_size; i += 8) {
        __m256 vec_vug = _mm256_loadu_ps(value_update_gate + i);
        __m256 vec_vpo = _mm256_loadu_ps(value_prev_out + i);
        __m256 vec_csv = _mm256_loadu_ps(cell_state_value + i);
        vec_vpo = _mm256_mul_ps(vec_vug, vec_vpo);
#ifdef __FMA__
        __m256 vec_out = _mm256_fmadd_ps(vec_csv, 
          _mm256_sub_ps(vec_one_256, vec_vug), vec_vpo);
#else
        __m256 vec_out = _mm256_add_ps(_mm256_mul_ps(vec_csv, 
          _mm256_sub_ps(vec_one_256, vec_vug)), vec_vpo);
#endif
        _mm256_storeu_ps(value_output + i, vec_out);
      }
  #endif
  #ifdef __SSE__
      for (; i + 3 < frame_size; i += 4) {
        __m128 vec_vug = _mm_loadu_ps(value_update_gate + i);
        __m128 vec_vpo = _mm_loadu_ps(value_prev_out + i);
        __m128 vec_csv = _mm_loadu_ps(cell_state_value + i);
        vec_vpo = _mm_mul_ps(vec_vug, vec_vpo);
        __m128 vec_out = _mm_add_ps(_mm_mul_ps(vec_csv, 
          _mm_sub_ps(vec_one_128, vec_vug)), vec_vpo);
        _mm_storeu_ps(value_output + i, vec_out);
      }
  #endif
      for (; i < frame_size; i++) {
        value_output[i] = (1.f - value_update_gate[i]) * cell_state_value[i] +
                           value_update_gate[i] * value_prev_out[i];
      }
    } else {
      i = 0;
  #ifdef __AVX__
      for (; i + 7 < frame_size; i += 8) {
        __m256 vec_vug = _mm256_loadu_ps(value_update_gate + i);
        __m256 vec_csv = _mm256_loadu_ps(cell_state_value + i);
        __m256 vec_out = _mm256_mul_ps(_mm256_sub_ps(vec_one_256, vec_vug), vec_csv);
        _mm256_storeu_ps(value_output + i, vec_out);
      }
  #endif
  #ifdef __SSE__
      for (; i + 3 < frame_size; i += 4) {
        __m128 vec_vug = _mm_loadu_ps(value_update_gate + i);
        __m128 vec_csv = _mm_loadu_ps(cell_state_value + i);
        __m128 vec_out = _mm_mul_ps(_mm_sub_ps(vec_one_128, vec_vug), vec_csv);
        _mm_storeu_ps(value_output + i, vec_out);
      }
  #endif
      for (; i < frame_size; i++) {
        value_output[i] = (1.f - value_update_gate[i]) * cell_state_value[i];
      }
    }

    value_reset_gate += frame_size * 3;
    value_update_gate += frame_size * 3;
    value_reset_output += frame_size;
    cell_state_value += frame_size * 3;
    value_output += frame_size;
    if (value.prev_out_value) {
      value_prev_out += frame_size;
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
