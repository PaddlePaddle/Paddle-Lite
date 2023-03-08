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

#include <memory>
#include <vector>

#include "lite/backends/arm/math/quantize.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

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
inline void gru_add_with_bias(
    const float* din, const float* bias, float* dout, int batch, int size) {
  LITE_PARALLEL_BEGIN(i, tid, batch) {
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
  LITE_PARALLEL_END()
}

template <lite_api::ActivationType Act>
static void gru_unit_reset_act_impl(float* updata_gate_ptr,
                                    int stride_update,
                                    float* reset_gate_ptr,
                                    int stride_reset,
                                    const float* hidden_prev_ptr,
                                    int stride_hidden_prev,
                                    float* reset_hidden_prev_ptr,
                                    int stride_reset_hidden_prev,
                                    int frame_size,
                                    int batch_size) {
  LITE_PARALLEL_BEGIN(b, tid, batch_size) {
    float* updata_gate = updata_gate_ptr;
    float* reset_gate = reset_gate_ptr;
    const float* hidden_prev = hidden_prev_ptr;
    float* reset_hidden_prev = reset_hidden_prev_ptr;

    updata_gate = updata_gate_ptr + b * stride_update;
    reset_gate = reset_gate_ptr + b * stride_reset;
    if (hidden_prev) {
      hidden_prev = hidden_prev_ptr + b * stride_hidden_prev;
    }
    reset_hidden_prev = reset_hidden_prev_ptr + b * stride_reset_hidden_prev;

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
  }
  LITE_PARALLEL_END()
}

template <lite_api::ActivationType Act>
static void gru_unit_out_act_impl(bool origin_mode,
                                  float* updata_gate_ptr,
                                  int stride_update,
                                  float* cell_state_ptr,
                                  int stride_cell_state,
                                  const float* hidden_prev_ptr,
                                  int stride_hidden_prev,
                                  float* hidden_ptr,
                                  int stride_hidden,
                                  int frame_size,
                                  int batch_size) {
  LITE_PARALLEL_BEGIN(b, tid, batch_size) {
    float* updata_gate = updata_gate_ptr;
    float* cell_state = cell_state_ptr;
    const float* hidden_prev = hidden_prev_ptr;
    float* hidden = hidden_ptr;

    updata_gate = updata_gate_ptr + b * stride_update;
    cell_state = cell_state_ptr + b * stride_cell_state;
    if (hidden_prev) {
      hidden_prev = hidden_prev_ptr + b * stride_hidden_prev;
    }
    hidden = hidden_ptr + b * stride_hidden;

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
  }
  LITE_PARALLEL_END()
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

    // Calculate W_{uh}h_{t-1} and W_{rh}h_{t-1}
    // Get u_{t} and r_{t} before applying activation
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

    // Get u_{t} and r_{t} after applying activation
    // Get r_{t}\odot h_{t-1}, save it to value.reset_output_value
    gru_unit_reset_act(active_gate, value, frame_size, batch_size);

    // Get W_{ch}(r_{t}\odot h_{t-1}), and it adds to cell_gate \hat{h_{t}}
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

    // Apply activation to cell_gate \hat{h_{t}} and get final h_{t}
    gru_unit_out_act(active_node, origin_mode, value, frame_size, batch_size);
  }

  static void quant_compute(GRUMetaValue<T> value,
                            int frame_size,
                            int batch_size,
                            const lite_api::ActivationType active_node,
                            const lite_api::ActivationType active_gate,
                            bool origin_mode,
                            std::vector<float> weight_scale,
                            int bit_length,
                            ARMContext* ctx) {
    operators::ActivationParam act_param;
    act_param.has_active = false;

    // Calculate W_{uh}h_{t-1} and W_{rh}h_{t-1}
    // Get u_{t} and r_{t} before applying activation
    if (value.prev_out_value) {
      // quantize h_{t-1}
      int prev_out_size = batch_size * frame_size;
      float prev_out_threshold =
          lite::arm::math::FindAbsMax(value.prev_out_value, prev_out_size);
      float prev_out_scale =
          lite::arm::math::GetScale(prev_out_threshold, bit_length);
      std::unique_ptr<int8_t[]> prev_out_value_int8(new int8_t[prev_out_size]);
      lite::arm::math::QuantizeTensor(value.prev_out_value,
                                      prev_out_value_int8.get(),
                                      prev_out_size,
                                      prev_out_scale);

      // update scale
      std::vector<float> scales(batch_size, weight_scale[0]);
      for (auto&& x : scales) {
        x *= prev_out_scale;
      }

      // gemm_s8
      std::unique_ptr<float[]> out_data(new float[batch_size * frame_size * 2]);
      lite::arm::math::gemm_s8(false,
                               false,
                               false,
                               batch_size,
                               frame_size * 2,
                               frame_size,
                               prev_out_value_int8.get(),
                               value.gate_weight_int8,
                               out_data.get(),
                               nullptr,
                               false,
                               lite::arm::math::GemmNoBias,
                               scales.data(),
                               act_param,
                               ctx);

      for (int i = 0; i < batch_size; i++) {
        float* dest = value.gate_value + i * frame_size * 3;
        float* src = out_data.get() + i * frame_size * 2;
        for (int j = 0; j < frame_size * 2; j++) {
          dest[j] += src[j];
        }
      }
    }

    // Get u_{t} and r_{t} after applying activation
    // Get r_{t}\odot h_{t-1}, save it to value.reset_output_value
    gru_unit_reset_act(active_gate, value, frame_size, batch_size);

    // Get W_{ch}(r_{t}\odot h_{t-1}), and it adds to cell_gate \hat{h_{t}}
    if (value.prev_out_value) {
      // quantize r_{t}\odot h_{t-1}
      int reset_out_size = batch_size * frame_size;
      float reset_out_threshold =
          lite::arm::math::FindAbsMax(value.reset_output_value, reset_out_size);
      float reset_out_scale =
          lite::arm::math::GetScale(reset_out_threshold, bit_length);
      std::unique_ptr<int8_t[]> reset_out_value_int8(
          new int8_t[reset_out_size]);
      lite::arm::math::QuantizeTensor(value.reset_output_value,
                                      reset_out_value_int8.get(),
                                      reset_out_size,
                                      reset_out_scale);

      std::vector<float> scales(batch_size, weight_scale[0]);
      for (auto&& x : scales) {
        x *= reset_out_scale;
      }

      std::unique_ptr<float[]> out_data(new float[batch_size * frame_size]);
      lite::arm::math::gemm_s8(false,
                               false,
                               false,
                               batch_size,
                               frame_size,
                               frame_size,
                               reset_out_value_int8.get(),
                               value.state_weight_int8,
                               out_data.get(),
                               nullptr,
                               false,
                               lite::arm::math::GemmNoBias,
                               scales.data(),
                               act_param,
                               ctx);

      for (int i = 0; i < batch_size; i++) {
        float* dest = value.gate_value + frame_size * 2 + i * frame_size * 3;
        float* src = out_data.get() + i * frame_size;
        for (int j = 0; j < frame_size; j++) {
          dest[j] += src[j];
        }
      }
    }

    // Apply activation to cell_gate \hat{h_{t}} and get final h_{t}
    gru_unit_out_act(active_node, origin_mode, value, frame_size, batch_size);
  }
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
