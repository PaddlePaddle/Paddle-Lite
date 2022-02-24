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

#include <memory>
#include <vector>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

template <typename T>
struct GRUMetaValue {
  T* gate_weight;         // W_{uh}h_{t-1} and W_{rh}h_{t-1}
  T* state_weight;        // W_{ch}
  T* gate_value;          // update_gate u_{t}, reset_gate r_{t} and cell_gate
                          // \hat{h_{t}}
  T* reset_output_value;  // r_{t}\odot h_{t-1}
  T* output_value;        // H_{t}
  T* prev_out_value;      // H_{t-1}

  int8_t* gate_weight_int8;   // int8_t W_{uh}h_{t-1} and W_{rh}h_{t-1}
  int8_t* state_weight_int8;  // int8_t W_{ch}
};

template <typename Dtype>
inline void gru_add_with_bias(
    const Dtype* din, const Dtype* bias, Dtype* dout, int batch, int size);

template <>
inline void gru_add_with_bias(const float16_t* din,
                              const float16_t* bias,
                              float16_t* dout,
                              int batch,
                              int size) {
  int cnt_32 = size >> 5;
  int rem_32 = size & 31;
  int cnt_8 = rem_32 >> 3;
  int rem_8 = rem_32 & 7;

  LITE_PARALLEL_BEGIN(i, tid, batch) {
    auto din_batch = din + i * size;
    auto bias_batch = bias;
    auto dout_batch = dout + i * size;
    for (int j = 0; j < cnt_32; j++) {
      float16x8_t vb0 = vld1q_f16(bias_batch);
      float16x8_t vin0 = vld1q_f16(din_batch);
      float16x8_t vb1 = vld1q_f16(bias_batch + 8);
      float16x8_t vin1 = vld1q_f16(din_batch + 8);
      float16x8_t vb2 = vld1q_f16(bias_batch + 16);
      float16x8_t vin2 = vld1q_f16(din_batch + 16);
      float16x8_t vb3 = vld1q_f16(bias_batch + 24);
      float16x8_t vin3 = vld1q_f16(din_batch + 24);
      float16x8_t vout0 = vaddq_f16(vb0, vin0);
      float16x8_t vout1 = vaddq_f16(vb1, vin1);
      float16x8_t vout2 = vaddq_f16(vb2, vin2);
      float16x8_t vout3 = vaddq_f16(vb3, vin3);
      bias_batch += 32;
      vst1q_f16(dout_batch, vout0);
      vst1q_f16(dout_batch + 8, vout1);
      din_batch += 32;
      vst1q_f16(dout_batch + 16, vout2);
      vst1q_f16(dout_batch + 24, vout3);
      dout_batch += 32;
    }
    for (int j = 0; j < cnt_8; j++) {
      float16x8_t vb0 = vld1q_f16(bias_batch);
      float16x8_t vin0 = vld1q_f16(din_batch);
      float16x8_t vout0 = vaddq_f16(vb0, vin0);
      bias_batch += 8;
      din_batch += 8;
      vst1q_f16(dout_batch, vout0);
      dout_batch += 8;
    }
    for (int j = 0; j < rem_8; j++) {
      *dout_batch = *din_batch + *bias_batch;
      din_batch++;
      bias_batch++;
      dout_batch++;
    }
  }
  LITE_PARALLEL_END()
}

template <lite_api::ActivationType Act>
static void gru_unit_reset_act_impl(float16_t* updata_gate,
                                    int stride_update,
                                    float16_t* reset_gate,
                                    int stride_reset,
                                    const float16_t* hidden_prev,
                                    int stride_hidden_prev,
                                    float16_t* reset_hidden_prev,
                                    int stride_reset_hidden_prev,
                                    int frame_size,
                                    int batch_size) {
  LITE_PARALLEL_BEGIN(b, tid, batch_size) {
    float16x8_t vpre0 = vdupq_n_f16(0.f);
    float16x8_t vpre1 = vdupq_n_f16(0.f);
    float16x8_t vpre2 = vdupq_n_f16(0.f);
    float16x8_t vpre3 = vdupq_n_f16(0.f);
    float16_t prev = 0.f;
    const float16_t* hidden_prev_ptr = hidden_prev;
    float16_t* updata_gate_ptr = updata_gate + b * stride_update;
    float16_t* reset_gate_ptr = reset_gate + b * stride_reset;
    if (hidden_prev) {
      hidden_prev_ptr = hidden_prev + b * stride_hidden_prev;
    }
    float16_t* reset_hidden_prev_ptr =
        reset_hidden_prev + b * stride_reset_hidden_prev;

    int i = 0;
    for (; i < frame_size - 31; i += 32) {
      float16x8_t vu0 = vld1q_f16(updata_gate_ptr + i);
      float16x8_t vu1 = vld1q_f16(updata_gate_ptr + i + 8);
      float16x8_t vu2 = vld1q_f16(updata_gate_ptr + i + 16);
      float16x8_t vu3 = vld1q_f16(updata_gate_ptr + i + 24);

      float16x8_t vr0 = vld1q_f16(reset_gate_ptr + i);
      float16x8_t vau0 = vactive_f16<Act>(vu0);
      float16x8_t vr1 = vld1q_f16(reset_gate_ptr + i + 8);
      float16x8_t vau1 = vactive_f16<Act>(vu1);
      float16x8_t vr2 = vld1q_f16(reset_gate_ptr + i + 16);
      float16x8_t vau2 = vactive_f16<Act>(vu2);
      float16x8_t vr3 = vld1q_f16(reset_gate_ptr + i + 24);
      float16x8_t vau3 = vactive_f16<Act>(vu3);

      if (hidden_prev) {
        vpre0 = vld1q_f16(hidden_prev_ptr + i);
        vpre1 = vld1q_f16(hidden_prev_ptr + i + 8);
        vpre2 = vld1q_f16(hidden_prev_ptr + i + 16);
        vpre3 = vld1q_f16(hidden_prev_ptr + i + 24);
      }
      float16x8_t var0 = vactive_f16<Act>(vr0);
      vst1q_f16(updata_gate_ptr + i, vau0);
      float16x8_t var1 = vactive_f16<Act>(vr1);
      vst1q_f16(updata_gate_ptr + i + 8, vau1);
      float16x8_t var2 = vactive_f16<Act>(vr2);
      vst1q_f16(updata_gate_ptr + i + 16, vau2);
      float16x8_t var3 = vactive_f16<Act>(vr3);
      vst1q_f16(updata_gate_ptr + i + 24, vau3);

      float16x8_t vres0 = vmulq_f16(vpre0, var0);
      vst1q_f16(reset_gate_ptr + i, var0);
      float16x8_t vres1 = vmulq_f16(vpre1, var1);
      vst1q_f16(reset_gate_ptr + i + 8, var1);
      float16x8_t vres2 = vmulq_f16(vpre2, var2);
      vst1q_f16(reset_gate_ptr + i + 16, var2);
      float16x8_t vres3 = vmulq_f16(vpre3, var3);
      vst1q_f16(reset_gate_ptr + i + 24, var3);

      vst1q_f16(reset_hidden_prev_ptr + i, vres0);
      vst1q_f16(reset_hidden_prev_ptr + i + 8, vres1);
      vst1q_f16(reset_hidden_prev_ptr + i + 16, vres2);
      vst1q_f16(reset_hidden_prev_ptr + i + 24, vres3);
    }
    for (; i < frame_size - 7; i += 8) {
      float16x8_t vu0 = vld1q_f16(updata_gate_ptr + i);
      float16x8_t vr0 = vld1q_f16(reset_gate_ptr + i);

      if (hidden_prev) {
        vpre0 = vld1q_f16(hidden_prev_ptr + i);
      }
      float16x8_t vau0 = vactive_f16<Act>(vu0);
      float16x8_t var0 = vactive_f16<Act>(vr0);

      float16x8_t vres0 = vmulq_f16(vpre0, var0);
      vst1q_f16(updata_gate_ptr + i, vau0);
      vst1q_f16(reset_gate_ptr + i, var0);
      vst1q_f16(reset_hidden_prev_ptr + i, vres0);
    }
    for (; i < frame_size; i++) {
      updata_gate_ptr[i] = active_f16<Act>(updata_gate_ptr[i]);
      reset_gate_ptr[i] = active_f16<Act>(reset_gate_ptr[i]);
      if (hidden_prev) {
        prev = hidden_prev_ptr[i];
      }
      reset_hidden_prev_ptr[i] = reset_gate_ptr[i] * prev;
    }
  }
  LITE_PARALLEL_END()
}

template <lite_api::ActivationType Act>
static void gru_unit_out_act_impl(bool origin_mode,
                                  float16_t* updata_gate,
                                  int stride_update,
                                  float16_t* cell_state,
                                  int stride_cell_state,
                                  const float16_t* hidden_prev,
                                  int stride_hidden_prev,
                                  float16_t* hidden,
                                  int stride_hidden,
                                  int frame_size,
                                  int batch_size) {
  if (origin_mode) {
    LITE_PARALLEL_BEGIN(b, tid, batch_size) {
      float16x8_t vpre0 = vdupq_n_f16(0.f);
      float16x8_t vpre1 = vdupq_n_f16(0.f);
      float16x8_t vpre2 = vdupq_n_f16(0.f);
      float16x8_t vpre3 = vdupq_n_f16(0.f);
      const float16_t* hidden_prev_ptr = hidden_prev;
      float16_t* updata_gate_ptr = updata_gate + b * stride_update;
      float16_t* cell_state_ptr = cell_state + b * stride_cell_state;
      if (hidden_prev) {
        hidden_prev_ptr = hidden_prev + b * stride_hidden_prev;
      }
      float16_t* hidden_ptr = hidden + b * stride_hidden;
      float16_t prev = 0.f;
      int i = 0;
      for (; i < frame_size - 31; i += 32) {
        float16x8_t vc0 = vld1q_f16(cell_state_ptr + i);
        float16x8_t vc1 = vld1q_f16(cell_state_ptr + i + 8);
        float16x8_t vc2 = vld1q_f16(cell_state_ptr + i + 16);
        float16x8_t vc3 = vld1q_f16(cell_state_ptr + i + 24);

        float16x8_t vu0 = vld1q_f16(updata_gate_ptr + i);
        float16x8_t vac0 = vactive_f16<Act>(vc0);
        float16x8_t vu1 = vld1q_f16(updata_gate_ptr + i + 8);
        float16x8_t vac1 = vactive_f16<Act>(vc1);
        float16x8_t vu2 = vld1q_f16(updata_gate_ptr + i + 16);
        float16x8_t vac2 = vactive_f16<Act>(vc2);
        float16x8_t vu3 = vld1q_f16(updata_gate_ptr + i + 24);
        float16x8_t vac3 = vactive_f16<Act>(vc3);
        if (hidden_prev) {
          vpre0 = vld1q_f16(hidden_prev_ptr + i);
          vpre1 = vld1q_f16(hidden_prev_ptr + i + 8);
          vpre2 = vld1q_f16(hidden_prev_ptr + i + 16);
          vpre3 = vld1q_f16(hidden_prev_ptr + i + 24);
        }

        float16x8_t vh0 = vfmsq_f16(vac0, vu0, vac0);
        float16x8_t vh1 = vfmsq_f16(vac1, vu1, vac1);
        float16x8_t vh2 = vfmsq_f16(vac1, vu2, vac2);
        float16x8_t vh3 = vfmsq_f16(vac1, vu3, vac3);

        vst1q_f16(cell_state_ptr + i, vac0);
        vh0 = vfmaq_f16(vh0, vu0, vpre0);
        vst1q_f16(cell_state_ptr + i + 8, vac1);
        vh1 = vfmaq_f16(vh1, vu1, vpre1);
        vst1q_f16(cell_state_ptr + i + 16, vac2);
        vh2 = vfmaq_f16(vh2, vu2, vpre2);
        vst1q_f16(cell_state_ptr + i + 24, vac3);
        vh3 = vfmaq_f16(vh3, vu3, vpre3);

        vst1q_f16(hidden_ptr + i, vh0);
        vst1q_f16(hidden_ptr + i + 8, vh1);
        vst1q_f16(hidden_ptr + i + 16, vh2);
        vst1q_f16(hidden_ptr + i + 24, vh3);
      }

      for (; i < frame_size - 7; i += 8) {
        float16x8_t vc0 = vld1q_f16(cell_state_ptr + i);
        float16x8_t vu0 = vld1q_f16(updata_gate_ptr + i);

        float16x8_t vac0 = vactive_f16<Act>(vc0);
        if (hidden_prev) {
          vpre0 = vld1q_f16(hidden_prev_ptr + i);
        }

        float16x8_t vh0 = vfmsq_f16(vac0, vu0, vac0);
        vst1q_f16(cell_state_ptr + i, vac0);
        vh0 = vfmaq_f16(vh0, vu0, vpre0);
        vst1q_f16(hidden_ptr + i, vh0);
      }
      for (; i < frame_size; i++) {
        if (hidden_prev) {
          prev = hidden_prev_ptr[i];
        }
        cell_state_ptr[i] = active_f16<Act>(cell_state_ptr[i]);
        hidden_ptr[i] = cell_state_ptr[i] * (1.f - updata_gate_ptr[i]) +
                        updata_gate_ptr[i] * prev;
      }
    }
    LITE_PARALLEL_END()
  } else {
    LITE_PARALLEL_BEGIN(b, tid, batch_size) {
      float16x8_t vpre0 = vdupq_n_f16(0.f);
      float16x8_t vpre1 = vdupq_n_f16(0.f);
      float16x8_t vpre2 = vdupq_n_f16(0.f);
      float16x8_t vpre3 = vdupq_n_f16(0.f);
      const float16_t* hidden_prev_ptr = hidden_prev;
      float16_t* updata_gate_ptr = updata_gate + b * stride_update;
      float16_t* cell_state_ptr = cell_state + b * stride_cell_state;
      if (hidden_prev) {
        hidden_prev_ptr = hidden_prev + b * stride_hidden_prev;
      }
      float16_t* hidden_ptr = hidden + b * stride_hidden;
      float16_t prev = 0.f;
      int i = 0;
      for (; i < frame_size - 31; i += 32) {
        float16x8_t vc0 = vld1q_f16(cell_state_ptr + i);
        float16x8_t vc1 = vld1q_f16(cell_state_ptr + i + 8);
        float16x8_t vc2 = vld1q_f16(cell_state_ptr + i + 16);
        float16x8_t vc3 = vld1q_f16(cell_state_ptr + i + 24);
        float16x8_t vu0 = vld1q_f16(updata_gate_ptr + i);

        float16x8_t vac0 = vactive_f16<Act>(vc0);
        float16x8_t vu1 = vld1q_f16(updata_gate_ptr + i + 8);
        float16x8_t vac1 = vactive_f16<Act>(vc1);
        float16x8_t vu2 = vld1q_f16(updata_gate_ptr + i + 16);
        float16x8_t vac2 = vactive_f16<Act>(vc2);
        float16x8_t vu3 = vld1q_f16(updata_gate_ptr + i + 24);
        float16x8_t vac3 = vactive_f16<Act>(vc3);
        if (hidden_prev) {
          vpre0 = vld1q_f16(hidden_prev_ptr + i);
          vpre1 = vld1q_f16(hidden_prev_ptr + i + 8);
          vpre2 = vld1q_f16(hidden_prev_ptr + i + 16);
          vpre3 = vld1q_f16(hidden_prev_ptr + i + 24);
        }
        float16x8_t vh0 = vfmsq_f16(vpre0, vpre0, vu0);
        float16x8_t vh1 = vfmsq_f16(vpre1, vpre1, vu1);
        float16x8_t vh2 = vfmsq_f16(vpre2, vpre2, vu2);
        float16x8_t vh3 = vfmsq_f16(vpre3, vpre3, vu3);

        vst1q_f16(cell_state_ptr + i, vac0);
        vh0 = vfmaq_f16(vh0, vu0, vac0);
        vst1q_f16(cell_state_ptr + i + 8, vac1);
        vh1 = vfmaq_f16(vh1, vu1, vac1);
        vst1q_f16(cell_state_ptr + i + 16, vac2);
        vh2 = vfmaq_f16(vh2, vu2, vac2);
        vst1q_f16(cell_state_ptr + i + 24, vac3);
        vh3 = vfmaq_f16(vh3, vu3, vac3);

        vst1q_f16(hidden_ptr + i, vh0);
        vst1q_f16(hidden_ptr + i + 8, vh1);
        vst1q_f16(hidden_ptr + i + 16, vh2);
        vst1q_f16(hidden_ptr + i + 24, vh3);
      }
      for (; i < frame_size - 7; i += 8) {
        float16x8_t vc0 = vld1q_f16(cell_state_ptr + i);
        float16x8_t vu0 = vld1q_f16(updata_gate_ptr + i);

        if (hidden_prev) {
          vpre0 = vld1q_f16(hidden_prev_ptr + i);
        }
        float16x8_t vac0 = vactive_f16<Act>(vc0);
        float16x8_t vh0 = vfmsq_f16(vpre0, vpre0, vu0);

        vst1q_f16(cell_state_ptr + i, vac0);
        vh0 = vfmaq_f16(vh0, vu0, vac0);
        vst1q_f16(hidden_ptr + i, vh0);
      }
      for (; i < frame_size; i++) {
        cell_state_ptr[i] = active_f16<Act>(cell_state_ptr[i]);
        if (hidden_prev) {
          prev = hidden_prev_ptr[i];
        }
        hidden_ptr[i] = prev * (1.f - updata_gate_ptr[i]) +
                        updata_gate_ptr[i] * cell_state_ptr[i];
      }
    }
    LITE_PARALLEL_END()
  }
}

inline void gru_unit_reset_act(lite_api::ActivationType act_type,
                               GRUMetaValue<float16_t> value,
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
                             GRUMetaValue<float16_t> value,
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

    // Calculate W_{uh}h_{t-1} and W_{rh}h_{t-1}
    // Get u_{t} and r_{t} before applying activation
    if (value.prev_out_value) {
      sgemm_fp16(false,
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

    // Get u_{t} and r_{t} after applying activation
    // Get r_{t}\odot h_{t-1}, save it to value.reset_output_value
    gru_unit_reset_act(active_gate, value, frame_size, batch_size);

    // Get W_{ch}(r_{t}\odot h_{t-1}), and it adds to cell_gate \hat{h_{t}}
    if (value.prev_out_value) {
      sgemm_fp16(false,
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

    // Apply activation to cell_gate \hat{h_{t}} and get final h_{t}
    gru_unit_out_act(active_node, origin_mode, value, frame_size, batch_size);
  }
};

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
