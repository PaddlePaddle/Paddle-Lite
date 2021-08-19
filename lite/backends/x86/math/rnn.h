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

#pragma once

#include <string>
#include "lite/backends/x86/math/blas.h"
#include "lite/core/tensor.h"
#include "lite/utils/logging.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <class T>
struct LstmMetaValue {
  T* gate_value;
  T* prev_state_value;
  T* state_value;
  T* state_active_value;
  T* output_value;
  T* check_ig;
  T* check_fg;
  T* check_og;
};

template <typename T>
struct GRUMetaValue {
  const T* gate_weight;
  const T* state_weight;
  const T* reset_bias;
  T* gate_value;
  T* reset_output_value;
  T* output_value;
  const T* prev_out_value;
};

// if v2 isn't null: out[i] = in[i] + v1[i] * v2[i];
// if v2 is null:    out[i] = in[i] * v1[i];
void vector_dot(float* out,
                const float* in,
                const float* v1,
                int size,
                const float* v2 = nullptr);

// only add bias
void fill_bias_fc(float* out, const float* bias, int num, int channel);

template <typename T>
void act_relu(const T* din, T* dout, int size, int threads);

template <typename T>
void act_sigmoid(const T* din, T* dout, int size, int threads);

template <typename T>
void act_tanh(const T* din, T* dout, int size, int threads);

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
struct RnnLstmUnitFunctor {
  static void compute(LstmMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      T cell_clip,
                      lite_api::ActivationType gate_act,
                      lite_api::ActivationType cell_act,
                      lite_api::ActivationType cand_act,
                      int threads) {
    const int temp_len = frame_size;
    auto zero_ptr = static_cast<float*>(
        TargetMalloc(TARGET(kX86), temp_len * sizeof(float)));
    memset(zero_ptr, 0, sizeof(float) * temp_len);

    for (int b = 0; b < batch_size; ++b) {
      T* value_ig = value.gate_value;
      T* value_fg = value_ig + frame_size;
      T* value_in = value_fg + frame_size;
      T* value_og = value_in + frame_size;
      T* state = value.state_value;
      T* state_act = value.state_active_value;

      T* check_i = value.check_ig ? value.check_ig : zero_ptr;
      T* check_f = value.check_fg ? value.check_fg : zero_ptr;
      T* check_o = value.check_og ? value.check_og : zero_ptr;
      T* prev_state =
          value.prev_state_value ? value.prev_state_value : zero_ptr;

      activation(value_in, value_in, frame_size, gate_act, threads);
      vector_dot(value_ig, value_ig, prev_state, frame_size, check_i);
      vector_dot(value_fg, value_fg, prev_state, frame_size, check_f);
      activation(value_ig, value_ig, frame_size, cell_act, threads);
      activation(value_fg, value_fg, frame_size, cell_act, threads);
      vector_dot(state, value_in, value_ig, frame_size);
      vector_dot(state, state, prev_state, frame_size, value_fg);

      for (int i = 0; i < frame_size; ++i) {
        if (cell_clip > 0.0) {
          if (state[i] < -1.0 * cell_clip) {
            state[i] = -1.0 * cell_clip;
          }
          if (state[i] > cell_clip) {
            state[i] = cell_clip;
          }
        }
      }

      vector_dot(value_og, value_og, state, frame_size, check_o);
      activation(value_og, value_og, frame_size, cell_act, threads);
      activation(state, state_act, frame_size, cand_act, threads);
      vector_dot(value.output_value, value_og, state_act, frame_size);

      value.gate_value += frame_size * 4;
      value.state_value += frame_size;
      value.state_active_value += frame_size;
      value.output_value += frame_size;
      if (value.prev_state_value) {
        value.prev_state_value += frame_size;
      }
    }

    TargetFree(TARGET(kX86), zero_ptr);
  }
};

template <typename T>
struct RnnGruUnitFunctorV2 {
  static void compute(X86Context* ctx,
                      GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      lite_api::ActivationType active_node,
                      lite_api::ActivationType active_gate) {
    if (value.prev_out_value) {
      lite::x86::math::Blas<lite::TargetType::kX86> matmul(*ctx);
      matmul.GEMM<float>(false,
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
                         frame_size);
    }

    auto value_reset_gate = value.gate_value;
    auto value_update_gate = value.gate_value + frame_size;
    auto value_reset_output = value.reset_output_value;
    auto value_reset_bias = value.reset_bias;
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
      }
      value_reset_gate += frame_size * 3;
      value_update_gate += frame_size * 3;
      value_reset_output += frame_size;
    }

    T* cell_state_value = value.gate_value + 2 * frame_size;
    T* reset_output_value = value.reset_output_value;
    for (int b = 0; b < batch_size; ++b) {
      for (int f = 0; f < frame_size; f++) {
        cell_state_value[f] += reset_output_value[f];
      }
      cell_state_value += frame_size * 3;
      reset_output_value += frame_size;
    }

    value_update_gate = value.gate_value + frame_size;
    auto value_frame_state = value.gate_value + 2 * frame_size;
    auto value_output = value.output_value;
    auto value_prev_out = value.prev_out_value;
    for (int b = 0; b < batch_size; b++) {
      activation(value_frame_state,
                 value_frame_state,
                 frame_size,
                 lite_api::ActivationType::kTanh_v2,
                 1);
      for (int i = 0; i < frame_size; i++) {
        value_output[i] = (1.f - value_update_gate[i]) * value_frame_state[i];
      }
      if (value.prev_out_value) {
        for (int i = 0; i < frame_size; i++) {
          value_output[i] =
              value_output[i] + value_update_gate[i] * value_prev_out[i];
        }
      }
      value_update_gate += frame_size * 3;
      value_frame_state += frame_size * 3;
      value_output += frame_size;
      if (value.prev_out_value) {
        value_prev_out += frame_size;
      }
    }
  }
};

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
