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
  const float* weight_data = weight->data<float>();
  const float* bias_data = nullptr;
  bool has_bias = false;
  if (bias) {
    bias_data = bias->data<float>();
    has_bias = true;
  }

  const float* hidden_prev_data = hidden_prev->data<float>();

  float* gate_data = gate->data<float>();
  float* reset_hidden_prev_data = resethiddenprev->data<float>();

  math::sgemm(hidden_prev_data,
              weight_data,
              bias_data,
              gate_data,
              batch_size,
              2 * frame_size,
              frame_size,
              has_bias,
              false,
              false,
              false,
              ctx);

  return;
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
