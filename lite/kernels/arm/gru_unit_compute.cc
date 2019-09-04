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
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/gru_utils.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

inline lite_api::ActivationType convert_gru_act_type(int act_type) {
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
    lite::arm::math::gru_add_with_bias<float>(
        input_data, bias_data, gate_data, batch_size, frame_size * 3);
  } else {
    for (int i = 0; i < batch_size; ++i) {
      TargetCopy(TargetType::kARM,
                 gate_data + i * frame_size * 3,
                 input_data,
                 frame_size * 3 * sizeof(float));
    }
  }

  lite::arm::math::GRUMetaValue<float> gru_value;
  gru_value.gate_weight = const_cast<float*>(weight_data);
  gru_value.state_weight =
      const_cast<float*>(weight_data + 2 * frame_size * frame_size);
  gru_value.prev_out_value = const_cast<float*>(hidden_prev_data);
  gru_value.output_value = hidden_data;
  gru_value.gate_value = gate_data;
  gru_value.reset_output_value = reset_hidden_prev_data;

  lite::arm::math::GRUUnitFunctor<float>::compute(
      gru_value,
      frame_size,
      batch_size,
      convert_gru_act_type(param.activation),
      convert_gru_act_type(param.gate_activation),
      param.origin_mode,
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
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("HiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Gate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
