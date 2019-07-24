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
#include <string>
#include <vector>
#include "lite/arm/math/funcs.h"
#include "lite/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

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
  }
}

void GRUUnitCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  // inputs
  auto input = param.input;
  auto hidden_prev = param.hiddenprev;
  auto weight = param.weight;
  auto bias = param.bias;
  // outputs
  auto gate = param.gate;
  auto resethiddenprev = param.resethiddenprev;
  auto hidden = param.hidden;

  int batch_size = input->dims()[0];
  int frame_size = hidden_prev->dims()[1];
  const float* input_data = input->data<float>();
  const float* hidden_prev_data = hidden_prev->data<float>();
  const float* weight_data = weight->data<float>();
  float* gate_data = gate->mutable_data<float>();
  float* reset_hidden_prev_data = resethiddenprev->mutable_data<float>();
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
                         0.f,
                         gate_data,
                         frame_size * 3,
                         nullptr,
                         false,
                         false,
                         &ctx);
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
    .BindInput("input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("hiddenprev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("gate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("resethiddenprev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
