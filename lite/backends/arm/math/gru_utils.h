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

#include "lite/backends/arm/math/sgemm.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
struct GRUMetaValue {
  T* gate_weight;
  T* state_weight;
  T* gate_value;
  T* reset_output_value;
  T* output_value;
  T* prev_out_value;
};

template <typename Dtype>
inline void gru_add_with_bias(
    const Dtype* din, const Dtype* bias, Dtype* dout, int batch, int size);

template <>
inline void gru_add_with_bias(
    const float* din, const float* bias, float* dout, int batch, int size) {
#pragma omp parallel for
  for (int i = 0; i < batch; ++i) {
    int j = 0;
    auto din_batch = din + i * size;
    auto dout_batch = dout + i * size;
    float32x4_t vb0 = vld1q_f32(bias);
    float32x4_t vin0 = vld1q_f32(din_batch);
    float32x4_t vout0;
    float32x4_t vout1;
    float32x4_t vin1;
    float32x4_t vb1;
    for (; j < size - 7; j += 8) {
      vin1 = vld1q_f32(din_batch + j + 4);
      vb1 = vld1q_f32(bias + j + 4);
      vout0 = vaddq_f32(vb0, vin0);
      vout1 = vaddq_f32(vb1, vin1);
      vb0 = vld1q_f32(bias + j + 8);
      vin0 = vld1q_f32(din_batch + j + 8);
      vst1q_f32(dout_batch + j, vout0);
      vst1q_f32(dout_batch + j + 4, vout1);
    }
    for (; j < size; ++j) {
      dout_batch[j] = din_batch[j] + bias[j];
    }
  }
}

template <lite_api::ActivationType Act>
static void gru_unit_reset_act_impl(float* updata_gate,
                                    int stride_update,
                                    float* reset_gate,
                                    int stride_reset,
                                    const float* hidden_prev,
                                    int stride_hidden_prev,
                                    float* reset_hidden_prev,
                                    int stride_reset_hidden_prev,
                                    int frame_size,
                                    int batch_size) {
#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    float32x4_t vpre0 = vdupq_n_f32(0.f);
    float32x4_t vpre1 = vdupq_n_f32(0.f);
    float prev = 0.f;
    int i = 0;
    for (; i < frame_size - 7; i += 8) {
      float32x4_t vu0 = vld1q_f32(updata_gate + i);
      float32x4_t vu1 = vld1q_f32(updata_gate + i + 4);
      float32x4_t vr0 = vld1q_f32(reset_gate + i);
      float32x4_t vr1 = vld1q_f32(reset_gate + i + 4);

      float32x4_t vau0 = lite::arm::math::vactive_f32<Act>(vu0);
      float32x4_t vau1 = lite::arm::math::vactive_f32<Act>(vu1);

      if (hidden_prev) {
        vpre0 = vld1q_f32(hidden_prev + i);
        vpre1 = vld1q_f32(hidden_prev + i + 4);
      }

      float32x4_t var0 = lite::arm::math::vactive_f32<Act>(vr0);
      float32x4_t var1 = lite::arm::math::vactive_f32<Act>(vr1);

      vst1q_f32(updata_gate + i, vau0);
      vst1q_f32(updata_gate + i + 4, vau1);

      float32x4_t vres0 = vmulq_f32(vpre0, var0);
      float32x4_t vres1 = vmulq_f32(vpre1, var1);

      vst1q_f32(reset_gate + i, var0);
      vst1q_f32(reset_gate + i + 4, var1);
      vst1q_f32(reset_hidden_prev + i, vres0);
      vst1q_f32(reset_hidden_prev + i + 4, vres1);
    }

    for (; i < frame_size; ++i) {
      updata_gate[i] = lite::arm::math::active_f32<Act>(updata_gate[i]);
      reset_gate[i] = lite::arm::math::active_f32<Act>(reset_gate[i]);
      if (hidden_prev) {
        prev = hidden_prev[i];
      }
      reset_hidden_prev[i] = reset_gate[i] * prev;
    }

    updata_gate += stride_update;
    reset_gate += stride_reset;
    if (hidden_prev) {
      hidden_prev += stride_hidden_prev;
    }
    reset_hidden_prev += stride_reset_hidden_prev;
  }
}

template <lite_api::ActivationType Act>
static void gru_unit_out_act_impl(bool origin_mode,
                                  float* updata_gate,
                                  int stride_update,
                                  float* cell_state,
                                  int stride_cell_state,
                                  const float* hidden_prev,
                                  int stride_hidden_prev,
                                  float* hidden,
                                  int stride_hidden,
                                  int frame_size,
                                  int batch_size) {
#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    float32x4_t vpre0 = vdupq_n_f32(0.f);
    float32x4_t vpre1 = vdupq_n_f32(0.f);
    float prev = 0.f;
    int i = 0;
    if (origin_mode) {
      for (; i < frame_size - 7; i += 8) {
        float32x4_t vc0 = vld1q_f32(cell_state + i);
        float32x4_t vc1 = vld1q_f32(cell_state + i + 4);
        float32x4_t vu0 = vld1q_f32(updata_gate + i);
        float32x4_t vu1 = vld1q_f32(updata_gate + i + 4);

        float32x4_t vac0 = lite::arm::math::vactive_f32<Act>(vc0);
        float32x4_t vac1 = lite::arm::math::vactive_f32<Act>(vc1);
        if (hidden_prev) {
          vpre0 = vld1q_f32(hidden_prev + i);
          vpre1 = vld1q_f32(hidden_prev + i + 4);
        }

        float32x4_t vh0 = vmlsq_f32(vac0, vu0, vac0);
        float32x4_t vh1 = vmlsq_f32(vac1, vu1, vac1);

        vst1q_f32(cell_state + i, vac0);
        vst1q_f32(cell_state + i + 4, vac1);

        vh0 = vmlaq_f32(vh0, vu0, vpre0);
        vh1 = vmlaq_f32(vh1, vu1, vpre1);

        vst1q_f32(hidden + i, vh0);
        vst1q_f32(hidden + i + 4, vh1);
      }

      for (; i < frame_size; ++i) {
        if (hidden_prev) {
          prev = hidden_prev[i];
        }
        cell_state[i] = lite::arm::math::active_f32<Act>(cell_state[i]);
        hidden[i] =
            cell_state[i] * (1.f - updata_gate[i]) + updata_gate[i] * prev;
      }
    } else {
      for (; i < frame_size - 7; i += 8) {
        float32x4_t vc0 = vld1q_f32(cell_state + i);
        float32x4_t vc1 = vld1q_f32(cell_state + i + 4);
        float32x4_t vu0 = vld1q_f32(updata_gate + i);
        float32x4_t vu1 = vld1q_f32(updata_gate + i + 4);

        float32x4_t vac0 = lite::arm::math::vactive_f32<Act>(vc0);
        float32x4_t vac1 = lite::arm::math::vactive_f32<Act>(vc1);

        if (hidden_prev) {
          vpre0 = vld1q_f32(hidden_prev + i);
          vpre1 = vld1q_f32(hidden_prev + i + 4);
        }

        float32x4_t vh0 = vmlsq_f32(vpre0, vpre0, vu0);
        float32x4_t vh1 = vmlsq_f32(vpre1, vpre1, vu1);

        vst1q_f32(cell_state + i, vac0);
        vst1q_f32(cell_state + i + 4, vac1);

        vh0 = vmlaq_f32(vh0, vu0, vac0);
        vh1 = vmlaq_f32(vh1, vu1, vac1);

        vst1q_f32(hidden + i, vh0);
        vst1q_f32(hidden + i + 4, vh1);
      }

      for (; i < frame_size; ++i) {
        cell_state[i] = lite::arm::math::active_f32<Act>(cell_state[i]);
        if (hidden_prev) {
          prev = hidden_prev[i];
        }
        hidden[i] =
            prev * (1.f - updata_gate[i]) + updata_gate[i] * cell_state[i];
      }
    }
    updata_gate += stride_update;
    cell_state += stride_cell_state;
    if (hidden_prev) {
      hidden_prev += stride_hidden_prev;
    }
    hidden += stride_hidden;
  }
}

inline void gru_unit_reset_act(lite_api::ActivationType act_type,
                               GRUMetaValue<float> value,
                               int frame_size,
                               int batch_size) {
  auto updata_gate = value.gate_value;
  auto reset_gate = value.gate_value + frame_size;
  auto hidden_prev = value.prev_out_value;
  auto reset_hidden_prev = value.reset_output_value;
  int stride_update = 3 * frame_size;
  int stride_reset = 3 * frame_size;
  int stride_hidden_prev = frame_size;
  int stride_reset_hidden_prev = frame_size;

  switch (act_type) {
    case lite_api::ActivationType::kIndentity:
      gru_unit_reset_act_impl<lite_api::ActivationType::kIndentity>(
          updata_gate,
          stride_update,
          reset_gate,
          stride_reset,
          hidden_prev,
          stride_hidden_prev,
          reset_hidden_prev,
          stride_reset_hidden_prev,
          frame_size,
          batch_size);
      break;
    case lite_api::ActivationType::kTanh:
      gru_unit_reset_act_impl<lite_api::ActivationType::kTanh>(
          updata_gate,
          stride_update,
          reset_gate,
          stride_reset,
          hidden_prev,
          stride_hidden_prev,
          reset_hidden_prev,
          stride_reset_hidden_prev,
          frame_size,
          batch_size);
      break;
    case lite_api::ActivationType::kSigmoid:
      gru_unit_reset_act_impl<lite_api::ActivationType::kSigmoid>(
          updata_gate,
          stride_update,
          reset_gate,
          stride_reset,
          hidden_prev,
          stride_hidden_prev,
          reset_hidden_prev,
          stride_reset_hidden_prev,
          frame_size,
          batch_size);
      break;
    case lite_api::ActivationType::kRelu:
      gru_unit_reset_act_impl<lite_api::ActivationType::kRelu>(
          updata_gate,
          stride_update,
          reset_gate,
          stride_reset,
          hidden_prev,
          stride_hidden_prev,
          reset_hidden_prev,
          stride_reset_hidden_prev,
          frame_size,
          batch_size);
      break;
    default:
      break;
  }
}

inline void gru_unit_out_act(lite_api::ActivationType act_type,
                             bool origin_mode,
                             GRUMetaValue<float> value,
                             int frame_size,
                             int batch_size) {
  auto updata_gate = value.gate_value;
  auto cell_state = value.gate_value + 2 * frame_size;
  auto hidden_prev = value.prev_out_value;
  auto hidden = value.output_value;

  int stride_update = 3 * frame_size;
  int stride_cell_state = 3 * frame_size;
  int stride_hidden_prev = frame_size;
  int stride_hidden = frame_size;

  switch (act_type) {
    case lite_api::ActivationType::kIndentity:
      gru_unit_out_act_impl<lite_api::ActivationType::kIndentity>(
          origin_mode,
          updata_gate,
          stride_update,
          cell_state,
          stride_cell_state,
          hidden_prev,
          stride_hidden_prev,
          hidden,
          stride_hidden,
          frame_size,
          batch_size);
      break;
    case lite_api::ActivationType::kTanh:
      gru_unit_out_act_impl<lite_api::ActivationType::kTanh>(origin_mode,
                                                             updata_gate,
                                                             stride_update,
                                                             cell_state,
                                                             stride_cell_state,
                                                             hidden_prev,
                                                             stride_hidden_prev,
                                                             hidden,
                                                             stride_hidden,
                                                             frame_size,
                                                             batch_size);
      break;
    case lite_api::ActivationType::kSigmoid:
      gru_unit_out_act_impl<lite_api::ActivationType::kSigmoid>(
          origin_mode,
          updata_gate,
          stride_update,
          cell_state,
          stride_cell_state,
          hidden_prev,
          stride_hidden_prev,
          hidden,
          stride_hidden,
          frame_size,
          batch_size);
      break;
    case lite_api::ActivationType::kRelu:
      gru_unit_out_act_impl<lite_api::ActivationType::kRelu>(origin_mode,
                                                             updata_gate,
                                                             stride_update,
                                                             cell_state,
                                                             stride_cell_state,
                                                             hidden_prev,
                                                             stride_hidden_prev,
                                                             hidden,
                                                             stride_hidden,
                                                             frame_size,
                                                             batch_size);
      break;
    default:
      break;
  }
}

template <typename T>
struct GRUUnitFunctor {
  static void compute(GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      const lite_api::ActivationType active_node,
                      const lite_api::ActivationType active_gate,
                      bool origin_mode,
                      ARMContext* ctx) {
    operators::ActivationParam act_param;
    act_param.has_active = false;
    if (value.prev_out_value) {
      sgemm(false,
            false,
            batch_size,
            frame_size * 2,
            frame_size,
            1.f,
            value.prev_out_value,
            frame_size,
            value.gate_weight,
            frame_size * 2,
            1.f,
            value.gate_value,
            frame_size * 3,
            nullptr,
            false,
            act_param,
            ctx);
    }
    gru_unit_reset_act(active_gate, value, frame_size, batch_size);

    if (value.prev_out_value) {
      sgemm(false,
            false,
            batch_size,
            frame_size,
            frame_size,
            1.f,
            value.reset_output_value,
            frame_size,
            value.state_weight,
            frame_size,
            1.f,
            value.gate_value + frame_size * 2,
            frame_size * 3,
            nullptr,
            false,
            act_param,
            ctx);
    }

    gru_unit_out_act(active_node, origin_mode, value, frame_size, batch_size);
  }
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
