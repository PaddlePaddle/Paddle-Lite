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

#include "lite/kernels/arm/gru_unit_compute.h"
#include "lite/api/paddle_place.h"
#include "lite/arm/math/funcs.h"
#include "lite/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/tests/kernels/test_funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

lite_api::ActivationType convert_gru_act_type(int act_type) {
  switch (act_type) {
    case 0:
      return lite_api::ActivationType::kIndentity;
    case 1:
      return lite_api::ActivationType::kSigmoid;
    case 2:
      return lite_api::ActivationType::kTanh;
    case 3:
      return lite_api::ActivationType::kRelu;
    default:
      return lite_api::ActivationType::kIndentity;
  }
}

template <typename Dtype>
void gru_add_with_bias(
    const Dtype* din, const Dtype* bias, Dtype* dout, int batch, int size);

template <>
void gru_add_with_bias(
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
void gru_unit_reset_act_impl(float* updata_gate,
                             int stride_update,
                             float* reset_gate,
                             int stride_reset,
                             const float* hidden_prev,
                             int stride_hidden_prev,
                             float* reset_hidden_prev,
                             int stride_reset_hidden_prev,
                             int frame_size,
                             int batch_size) {
  for (int b = 0; b < batch_size; ++b) {
    int i = 0;
    for (; i < frame_size - 7; i += 8) {
      float32x4_t vu0 = vld1q_f32(updata_gate + i);
      float32x4_t vu1 = vld1q_f32(updata_gate + i + 4);
      float32x4_t vr0 = vld1q_f32(reset_gate + i);
      float32x4_t vr1 = vld1q_f32(reset_gate + i + 4);

      float32x4_t vau0 = lite::arm::math::vactive_f32<Act>(vu0);
      float32x4_t vau1 = lite::arm::math::vactive_f32<Act>(vu1);

      float32x4_t vpre0 = vld1q_f32(hidden_prev + i);
      float32x4_t vpre1 = vld1q_f32(hidden_prev + i + 4);

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
      reset_hidden_prev[i] = reset_gate[i] * hidden_prev[i];
    }

    updata_gate += stride_update;
    reset_gate += stride_reset;
    hidden_prev += stride_hidden_prev;
    reset_hidden_prev += stride_reset_hidden_prev;
  }
}

template <lite_api::ActivationType Act>
void gru_unit_out_act_impl(bool origin_mode,
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
    int i = 0;
    if (origin_mode) {
      for (; i < frame_size - 7; i += 8) {
        float32x4_t vc0 = vld1q_f32(cell_state + i);
        float32x4_t vc1 = vld1q_f32(cell_state + i + 4);
        float32x4_t vu0 = vld1q_f32(updata_gate + i);
        float32x4_t vu1 = vld1q_f32(updata_gate + i + 4);

        float32x4_t vac0 = lite::arm::math::vactive_f32<Act>(vc0);
        float32x4_t vac1 = lite::arm::math::vactive_f32<Act>(vc1);

        float32x4_t vpre0 = vld1q_f32(hidden_prev + i);
        float32x4_t vpre1 = vld1q_f32(hidden_prev + i + 4);

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
        cell_state[i] = lite::arm::math::active_f32<Act>(cell_state[i]);
        hidden[i] = cell_state[i] * (1.f - updata_gate[i]) +
                    updata_gate[i] * hidden_prev[i];
      }
    } else {
      for (; i < frame_size - 7; i += 8) {
        float32x4_t vc0 = vld1q_f32(cell_state + i);
        float32x4_t vc1 = vld1q_f32(cell_state + i + 4);
        float32x4_t vu0 = vld1q_f32(updata_gate + i);
        float32x4_t vu1 = vld1q_f32(updata_gate + i + 4);

        float32x4_t vac0 = lite::arm::math::vactive_f32<Act>(vc0);
        float32x4_t vac1 = lite::arm::math::vactive_f32<Act>(vc1);

        float32x4_t vpre0 = vld1q_f32(hidden_prev + i);
        float32x4_t vpre1 = vld1q_f32(hidden_prev + i + 4);

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
        hidden[i] = hidden_prev[i] * (1.f - updata_gate[i]) +
                    updata_gate[i] * cell_state[i];
      }
    }
    updata_gate += stride_update;
    cell_state += stride_cell_state;
    hidden_prev += stride_hidden_prev;
    hidden += stride_hidden;
  }
}

void gru_unit_reset_act(lite_api::ActivationType act_type,
                        float* updata_gate,
                        int stride_update,
                        float* reset_gate,
                        int stride_reset,
                        const float* hidden_prev,
                        int stride_hidden_prev,
                        float* reset_hidden_prev,
                        int stride_reset_hidden_prev,
                        int frame_size,
                        int batch_size) {
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

void gru_unit_out_act(lite_api::ActivationType act_type,
                      bool origin_mode,
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

void GRUUnitCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  // inputs
  auto input = param.input;
  auto hidden_prev = param.hidden_prev;
  auto weight = param.weight;
  auto bias = param.bias;
  // outputs
  auto gate = param.gate;
  auto reset_hidden_prev = param.reset_hidden_prev;
  auto hidden = param.hidden;

  int batch_size = input->dims()[0];
  int frame_size = hidden_prev->dims()[1];
  const float* input_data = input->data<float>();
  const float* hidden_prev_data = hidden_prev->data<float>();
  const float* weight_data = weight->data<float>();
  float* gate_data = gate->mutable_data<float>();
  float* reset_hidden_prev_data = reset_hidden_prev->mutable_data<float>();
  float* hidden_data = hidden->mutable_data<float>();

  if (bias) {
    auto bias_data = bias->data<float>();
    gru_add_with_bias<float>(
        input_data, bias_data, gate_data, batch_size, frame_size * 3);
  } else {
    for (int i = 0; i < batch_size; ++i) {
      TargetCopy(TargetType::kARM,
                 gate_data + i * frame_size * 3,
                 input_data,
                 frame_size * 3 * sizeof(float));
    }
  }

  lite::arm::math::sgemm(false,
                         false,
                         batch_size,
                         2 * frame_size,
                         frame_size,
                         1.f,
                         hidden_prev_data,
                         frame_size,
                         weight_data,
                         frame_size * 2,
                         1.f,
                         gate_data,
                         frame_size * 3,
                         nullptr,
                         false,
                         false,
                         &ctx);

  gru_unit_reset_act(convert_gru_act_type(param.gate_activation),
                     gate_data,
                     3 * frame_size,
                     gate_data + frame_size,
                     3 * frame_size,
                     hidden_prev_data,
                     frame_size,
                     reset_hidden_prev_data,
                     frame_size,
                     frame_size,
                     batch_size);

  lite::arm::math::sgemm(false,
                         false,
                         batch_size,
                         frame_size,
                         frame_size,
                         1.f,
                         reset_hidden_prev_data,
                         frame_size,
                         weight_data + 2 * frame_size * frame_size,
                         frame_size,
                         1.f,
                         gate_data + frame_size * 2,
                         frame_size * 3,
                         nullptr,
                         false,
                         false,
                         &ctx);

  gru_unit_out_act(convert_gru_act_type(param.activation),
                   param.origin_mode,
                   gate_data,
                   3 * frame_size,
                   gate_data + 2 * frame_size,
                   3 * frame_size,
                   hidden_prev_data,
                   frame_size,
                   hidden_data,
                   frame_size,
                   frame_size,
                   batch_size);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gru_unit,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GRUUnitCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("HiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Gate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
