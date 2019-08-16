/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef GRU_OP

#pragma once

#include <type_traits>
#include "operators/math/activation.h"
#include "operators/math/gru_compute.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <typename T, ActivationType Act>
void hl_naive_gru_forward_reset_output(T *gate_value, T *reset_output_value,
                                       T *prev_output_value, int frame_size) {
  T r_value_update_gate;
  T r_value_reset_gate;
  T r_value_reset_output;
  T r_prev_out = 0;
  T *update_gate = gate_value;
  T *reset_gate = gate_value + frame_size;

  int remain = frame_size;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  int loop = remain >> 3;
  remain = remain & 0x7;
  float32x4_t prev0 = vdupq_n_f32(0.f);
  float32x4_t prev1 = vdupq_n_f32(0.f);
  for (int i = 0; i < loop; ++i) {
    float32x4_t update0 = vld1q_f32(update_gate);
    float32x4_t update1 = vld1q_f32(update_gate + 4);
    float32x4_t reset0 = vld1q_f32(reset_gate);
    float32x4_t reset1 = vld1q_f32(reset_gate + 4);
    if (prev_output_value) {
      prev0 = vld1q_f32(prev_output_value);
      prev1 = vld1q_f32(prev_output_value + 4);
      prev_output_value += 8;
    }
    update0 = vActiveq_f32<Act>(update0);
    update1 = vActiveq_f32<Act>(update1);
    reset0 = vActiveq_f32<Act>(reset0);
    reset1 = vActiveq_f32<Act>(reset1);
    float32x4_t output0 = vmulq_f32(prev0, reset0);
    float32x4_t output1 = vmulq_f32(prev1, reset1);
    vst1q_f32(update_gate, update0);
    vst1q_f32(update_gate + 4, update1);
    vst1q_f32(reset_gate, reset0);
    vst1q_f32(reset_gate + 4, reset1);
    vst1q_f32(reset_output_value, output0);
    vst1q_f32(reset_output_value + 4, output1);
    update_gate += 8;
    reset_gate += 8;
    reset_output_value += 8;
  }
#endif  // __ARM_NEON__
  for (int i = 0; i < remain; i++) {
    r_value_update_gate = update_gate[i];
    r_value_reset_gate = reset_gate[i];
    if (prev_output_value) {
      r_prev_out = prev_output_value[i];
    }
    r_value_update_gate = Active<Act>(r_value_update_gate);
    r_value_reset_gate = Active<Act>(r_value_reset_gate);
    r_value_reset_output = r_prev_out * r_value_reset_gate;
    update_gate[i] = r_value_update_gate;
    reset_gate[i] = r_value_reset_gate;
    reset_output_value[i] = r_value_reset_output;
  }
}

template <typename T, ActivationType Act>
void hl_naive_gru_forward_final_output(T *gate_value, T *prev_output_value,
                                       T *output_value, int frame_size) {
  T r_value_update_gate;
  T r_value_frame_state;
  T r_prev_out = 0;
  T r_output;
  T *update_gate = gate_value;
  T *frame_state = gate_value + frame_size * 2;

  int remain = frame_size;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  int loop = remain >> 3;
  remain = remain & 0x7;
  float32x4_t prev0 = vdupq_n_f32(0.f);
  float32x4_t prev1 = vdupq_n_f32(0.f);
  for (int i = 0; i < loop; ++i) {
    float32x4_t update0 = vld1q_f32(update_gate);
    float32x4_t update1 = vld1q_f32(update_gate + 4);
    float32x4_t state0 = vld1q_f32(frame_state);
    float32x4_t state1 = vld1q_f32(frame_state + 4);
    if (prev_output_value) {
      prev0 = vld1q_f32(prev_output_value);
      prev1 = vld1q_f32(prev_output_value + 4);
      prev_output_value += 8;
    }
    state0 = vActiveq_f32<Act>(state0);
    state1 = vActiveq_f32<Act>(state1);
    float32x4_t output0 = vmlsq_f32(prev0, update0, prev0);
    float32x4_t output1 = vmlsq_f32(prev1, update1, prev1);
    output0 = vmlaq_f32(output0, update0, state0);
    output1 = vmlaq_f32(output1, update1, state1);
    vst1q_f32(frame_state, state0);
    vst1q_f32(frame_state + 4, state1);
    vst1q_f32(output_value, output0);
    vst1q_f32(output_value + 4, output1);
    update_gate += 8;
    frame_state += 8;
    output_value += 8;
  }
#endif  // __ARM_NEON__
  for (int i = 0; i < remain; i++) {
    r_value_update_gate = update_gate[i];
    r_value_frame_state = frame_state[i];
    if (prev_output_value) {
      r_prev_out = prev_output_value[i];
    }
    r_value_frame_state = Active<Act>(r_value_frame_state);
    r_output = r_prev_out - r_value_update_gate * r_prev_out +
               r_value_update_gate * r_value_frame_state;
    frame_state[i] = r_value_frame_state;
    output_value[i] = r_output;
  }
}

#define FORWARD_RESET_OUTPUT(active_type, value, frame_size)            \
  hl_naive_gru_forward_reset_output<float, active_type>(                \
      value.gate_value, value.reset_output_value, value.prev_out_value, \
      frame_size);

template <typename T>
inline void forward_reset_output(GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_node) {
  for (int b = 0; b < batch_size; ++b) {
    switch (active_node) {
      case RELU:
        FORWARD_RESET_OUTPUT(RELU, value, frame_size);
        break;
      case SIGMOID:
        FORWARD_RESET_OUTPUT(SIGMOID, value, frame_size);
        break;
      case TANH:
        FORWARD_RESET_OUTPUT(TANH, value, frame_size);
        break;
      default:
        FORWARD_RESET_OUTPUT(IDENTITY, value, frame_size);
    }
    value.gate_value += frame_size * 3;
    value.reset_output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

#define FORWARD_FINAL_OUTPUT(active_type, value, frame_size) \
  hl_naive_gru_forward_final_output<float, active_type>(     \
      value.gate_value, value.prev_out_value, value.output_value, frame_size)

template <typename T>
inline void forward_final_output(GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_node) {
  for (int b = 0; b < batch_size; ++b) {
    switch (active_node) {
      case RELU:
        FORWARD_FINAL_OUTPUT(RELU, value, frame_size);
        break;
      case SIGMOID:
        FORWARD_FINAL_OUTPUT(SIGMOID, value, frame_size);
        break;
      case TANH:
        FORWARD_FINAL_OUTPUT(TANH, value, frame_size);
        break;
      default:
        FORWARD_FINAL_OUTPUT(IDENTITY, value, frame_size);
    }
    value.gate_value += frame_size * 3;
    value.output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
