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

#include <arm_neon.h>
#include <string>
#include "lite/backends/arm/math/activation.h"
#include "lite/core/tensor.h"
#include "lite/utils/logging.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {

void add_bias_rowwise(Tensor* input,
                      const Tensor* bias,
                      int start_w,
                      int end_w);

inline float* row_offset(Tensor& input, int start) {  // NOLINT
  auto in_dim = input.dims();
  int width = input.numel() / in_dim[0];
  int offset = start < in_dim[0] ? start * width : input.numel();
  return input.mutable_data<float>() + offset;
}
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

void vector_dot(float* out,
                const float* in,
                const float* v1,
                int size,
                const float* v2 = nullptr);

template <typename T>
struct LstmUnitFunctor {
  static void compute(LstmMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      T cell_clip,
                      std::string gate_act,
                      std::string cell_act,
                      std::string cand_act,
                      int threads) {
    for (int b = 0; b < batch_size; ++b) {
      const int temp_len = frame_size;
      float zero_ptr[temp_len];  // NOLINT
      memset(zero_ptr, 0, sizeof(float) * temp_len);

      T* value_in = value.gate_value;
      T* value_ig = value_in + frame_size;
      T* value_fg = value_ig + frame_size;
      T* value_og = value_fg + frame_size;
      T* state = value.state_value;
      T* state_act = value.state_active_value;
      T* output = value.output_value;

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
  }
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
