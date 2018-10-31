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
#include "operators/math/activation_functions.h"
#include "operators/math/gru_compute.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <class OpResetOutput, typename T>
void hl_naive_gru_forward_reset_output(OpResetOutput op_reset_output,
                                       T *gate_value, T *reset_output_value,
                                       T *prev_output_value, int frame_size,
                                       ActivationType active_gate) {
  T r_value_update_gate;
  T r_value_reset_gate;
  T r_value_reset_output;
  T r_prev_out = 0;
  T *update_gate = gate_value;
  T *reset_gate = gate_value + frame_size;

  for (int i = 0; i < frame_size; i++) {
    r_value_update_gate = update_gate[i];
    r_value_reset_gate = reset_gate[i];
    if (prev_output_value) {
      r_prev_out = prev_output_value[i];
    }

    op_reset_output(&r_value_update_gate, &r_value_reset_gate, &r_prev_out,
                    &r_value_reset_output, active_gate);

    update_gate[i] = r_value_update_gate;
    reset_gate[i] = r_value_reset_gate;
    reset_output_value[i] = r_value_reset_output;
  }
}

template <class OpFinalOutput, typename T>
void hl_naive_gru_forward_final_output(OpFinalOutput op_final_output,
                                       T *gate_value, T *prev_output_value,
                                       T *output_value, int frame_size,
                                       ActivationType active_node) {
  T r_value_update_gate;
  T r_value_frame_state;
  T r_prev_out = 0;
  T r_output;
  T *update_gate = gate_value;
  T *frame_state = gate_value + frame_size * 2;

  for (int i = 0; i < frame_size; i++) {
    r_value_update_gate = update_gate[i];
    r_value_frame_state = frame_state[i];
    if (prev_output_value) {
      r_prev_out = prev_output_value[i];
    }

    op_final_output(&r_value_update_gate, &r_value_frame_state, &r_prev_out,
                    &r_output, active_node);

    frame_state[i] = r_value_frame_state;
    output_value[i] = r_output;
  }
}

template <class OpResetOutput, typename T>
inline void forward_reset_output(OpResetOutput op_reset_output,
                                 GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_gate) {
  for (int b = 0; b < batch_size; b++) {
    hl_naive_gru_forward_reset_output(
        op_reset_output, value.gate_value, value.reset_output_value,
        value.prev_out_value, frame_size, active_gate);

    value.gate_value += frame_size * 3;
    value.reset_output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

template <class OpFinalOutput, typename T>
inline void forward_final_output(OpFinalOutput op_final_output,
                                 GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_node) {
  for (int b = 0; b < batch_size; b++) {
    hl_naive_gru_forward_final_output(op_final_output, value.gate_value,
                                      value.prev_out_value, value.output_value,
                                      frame_size, active_node);

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
