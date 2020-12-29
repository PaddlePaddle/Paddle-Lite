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

#include "lite/kernels/arm/rnn_compute.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/lstm.h"
#include "lite/backends/arm/math/sequence2batch.h"
#include "lite/backends/arm/math/sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void RnnCompute::Run() {
  auto& param = this->Param<operators::RnnParam>();
  std::string mode = param.mode;
  auto& ctx = this->ctx_->As<ARMContext>();
  auto input = param.Input;
  auto weight_list = param.WeightList;
  auto reserve = param.Reserve;
  auto output = param.Out;
  bool is_bidirec = param.is_bidirec;
  int num_layers = param.num_layers;

  for (int i = 0; i < num_layers; i++) {
    auto* i_data = input->data<float>();
    auto* gate_data = reserve->mutable_data<float>();
    auto* w_data = weight_list[0]->data<float>();
    auto* o_data = output->data<float>();

    const float* b_data =
        weight_list[8] ? weight_list[8]->data<float>() : nullptr;

    bool flag_act = false;
    operators::ActivationParam act_param;
    act_param.has_active = false;
    auto input_dims = input->dims();
    auto weight_input_dims = weight_list[0]->dims();
    auto bias_input_dims = weight_list[8]->dims();
    auto weight_hidden_dims = weight_list[1]->dims();

    int m = input_dims[0] * input_dims[1];
    int k = input_dims[2];
    int n = weight_input_dims[0];

    lite::arm::math::sgemm(false,
                           true,
                           m,
                           n,
                           k,
                           1.f,
                           i_data,
                           k,
                           w_data,
                           n,
                           0.f,
                           gate_data,
                           n,
                           nullptr,
                           false,
                           act_param,
                           &ctx);
    if (b_data) {
      CHECK_EQ(bias_input_dims[0], n);
      // lite::arm::math::fill_bias_fc(o_data, b_data, m, n, flag_act);
    }
    /*
    lite::arm::math::LstmCell::compute(float* gateData, float* hidden_weight,
    float* hidden_bias, float* cellData, int hidden_size, int seq, float*
    output);
    */
    // if (is_bidirec) {
    //}
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    rnn, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::RnnCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("WeightList", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("PreState", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("SequenceLength", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("DropoutState", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Reserve", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("State", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
