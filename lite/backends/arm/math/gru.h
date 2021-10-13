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

#pragma once

#include "lite/backends/arm/math/sgemm.h"
#ifdef LITE_WITH_ARM
#include <arm_neon.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
struct RNNGRUValue {
  const T* gate_weight;
  const T* state_weight;
  const T* reset_bias;
  T* gate_value;
  T* reset_output_value;
  T* output_value;
  const T* prev_out_value;
};

template <typename T>
void rnn_activation(const T* din,
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
void compute_kernel(RNNGRUValue<T> value,
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
    rnn_activation(value_reset_gate,
                   value_reset_gate,
                   frame_size,
                   lite_api::ActivationType::kSigmoid_v2,
                   1);
    rnn_activation(value_update_gate,
                   value_update_gate,
                   frame_size,
                   lite_api::ActivationType::kSigmoid_v2,
                   1);

    for (int i = 0; i < frame_size; i++) {
      value_reset_output[i] =
          (value_reset_output[i] + value_reset_bias[i]) * value_reset_gate[i];
      cell_state_value[i] += value_reset_output[i];
    }

    rnn_activation(cell_state_value,
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
void compute_kernel<float>(RNNGRUValue<float> value,
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
  float32x4_t vec_one = vdupq_n_f32(1.f);

  for (int b = 0; b < batch_size; b++) {
    rnn_activation(value_reset_gate,
                   value_reset_gate,
                   frame_size,
                   lite_api::ActivationType::kSigmoid_v2,
                   1);
    rnn_activation(value_update_gate,
                   value_update_gate,
                   frame_size,
                   lite_api::ActivationType::kSigmoid_v2,
                   1);

    for (i = 0; i + 3 < frame_size; i += 4) {
      float32x4_t vec_out = vld1q_f32(value_reset_output + i);
      float32x4_t vec_reset = vld1q_f32(value_reset_gate + i);
      float32x4_t vec_bias = vld1q_f32(value_reset_bias + i);
      vec_out = vmulq_f32(vaddq_f32(vec_out, vec_bias), vec_reset);
      vst1q_f32(value_reset_output + i, vec_out);
      vst1q_f32(cell_state_value + i,
                vaddq_f32(vec_out, vld1q_f32(cell_state_value + i)));
    }
    for (; i < frame_size; i++) {
      value_reset_output[i] =
          (value_reset_output[i] + value_reset_bias[i]) * value_reset_gate[i];
      cell_state_value[i] += value_reset_output[i];
    }

    rnn_activation(cell_state_value,
                   cell_state_value,
                   frame_size,
                   lite_api::ActivationType::kTanh_v2,
                   1);

    if (value.prev_out_value) {
      for (i = 0; i + 3 < frame_size; i += 4) {
        float32x4_t vec_vug = vld1q_f32(value_update_gate + i);
        float32x4_t vec_vpo = vld1q_f32(value_prev_out + i);
        float32x4_t vec_csv = vld1q_f32(cell_state_value + i);
        vec_vpo = vmulq_f32(vec_vug, vec_vpo);
        float32x4_t vec_out =
            vmlaq_f32(vec_vpo, vsubq_f32(vec_one, vec_vug), vec_csv);
        vst1q_f32(value_output + i, vec_out);
      }
      for (; i < frame_size; i++) {
        value_output[i] = (1.f - value_update_gate[i]) * cell_state_value[i] +
                          value_update_gate[i] * value_prev_out[i];
      }
    } else {
      for (i = 0; i + 3 < frame_size; i += 4) {
        float32x4_t vec_vug = vld1q_f32(value_update_gate + i);
        float32x4_t vec_csv = vld1q_f32(cell_state_value + i);
        float32x4_t vec_out = vmulq_f32(vsubq_f32(vec_one, vec_vug), vec_csv);
        vst1q_f32(value_output + i, vec_out);
      }
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

template <typename T>
struct RnnGruUnitFunctorV2 {
  static void compute(ARMContext* ctx,
                      RNNGRUValue<T> value,
                      int frame_size,
                      int batch_size,
                      lite_api::ActivationType active_node,
                      lite_api::ActivationType active_gate) {
    if (value.prev_out_value) {
      operators::ActivationParam act_param;
      act_param.has_active = false;
      lite::arm::math::sgemm(false,
                             true,
                             batch_size,
                             frame_size,
                             frame_size,
                             1.f,
                             value.prev_out_value,
                             frame_size,
                             value.state_weight,
                             frame_size,
                             0.f,
                             value.reset_output_value,
                             frame_size,
                             nullptr,
                             false,
                             act_param,
                             ctx);
    }
    compute_kernel(value, frame_size, batch_size, active_node, active_gate);
  }
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
