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

template <typename TensorType>
void SplitReserveData(ARMContext* ctx,
                      TensorType* reserve_data,
                      Tensor* gate_data,
                      Tensor* cell_data,
                      Tensor* cell_act_data,
                      Tensor* hidden_data,
                      int direction_num,
                      const int& time_step,
                      const int& batch_size,
                      const int& hidden_size,
                      const int& gate_num,
                      const int& num_layers) {
  const int& gate_data_idx = gate_num * num_layers;
  const int& cell_data_idx = (gate_num + 1) * num_layers;
  const int& cell_act_data_idx = (gate_num + 2) * num_layers;
  // simple rnn
  int hidden_data_start_idx = gate_data_idx;
  *gate_data = reserve_data->Slice(0, gate_data_idx);
  *cell_data = reserve_data->Slice(gate_data_idx, cell_data_idx);
  *cell_act_data = reserve_data->Slice(cell_data_idx, cell_act_data_idx);
  hidden_data_start_idx = cell_act_data_idx;
  int hidden_data_idx = hidden_data_start_idx + (num_layers - 1);
  if (num_layers > 1) {
    *hidden_data = reserve_data->Slice(hidden_data_start_idx, hidden_data_idx);
  }
}

template <typename T>
void AllocateReserveData(ARMContext* ctx,
                         Tensor* reserve_data,
                         Tensor* gate_data,
                         Tensor* cell_data,
                         Tensor* cell_act_data,
                         Tensor* hidden_data,
                         const Tensor* input,
                         bool is_bidirec,
                         int num_layers,
                         int gate_num,
                         int hidden_size) {
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& time_step = input->dims()[0];
  const int& batch_size = input->dims()[1];
  const int& block_size = direction_num * time_step * batch_size * hidden_size;
  int hidden_data_idx = (num_layers - 1);
  hidden_data_idx += (gate_num + 2) * num_layers;
  reserve_data->Resize({hidden_data_idx, block_size});
  reserve_data->mutable_data<T>();
  SplitReserveData(ctx,
                   reserve_data,
                   gate_data,
                   cell_data,
                   cell_act_data,
                   hidden_data,
                   direction_num,
                   time_step,
                   batch_size,
                   hidden_size,
                   gate_num,
                   num_layers);
}

void RnnCompute::Run() {
  auto& param = this->Param<operators::RnnParam>();
  std::string mode = param.mode;
  auto& ctx = this->ctx_->As<ARMContext>();
  auto input = param.Input;
  auto weight_list = param.WeightList;
  auto reserve_data = param.Reserve;
  auto output = param.Out;
  bool is_bidirec = param.is_bidirec;
  int num_layers = param.num_layers;

  Tensor gate_data, cell_data, cell_act_data, hidden_data;
  AllocateReserveData<float>(ctx,
                             reserve_data,
                             &gate_data,
                             &cell_data,
                             &cell_act_data,
                             &hidden_data,
                             input,
                             is_bidirec,
                             num_layers,
                             gate_num,
                             hidden_size);
  gate_data.Resize({num_layers, gate_data.numel() / num_layers});
  cell_data.Resize({num_layers, cell_data.numel() / num_layers});
  cell_act_data.Resize({num_layers, cell_act_data.numel() / num_layers});
  if (num_layers > 1) {
    hidden_data.Resize(
        {num_layers - 1, hidden_data.numel() / (num_layers - 1)});
  }

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
      lite::arm::math::fill_bias_fc(o_data, b_data, m, n, flag_act);
    }

    lstm_value.gate_value = gate_t;
    lstm_value.output_value = out_t;
    lstm_value.state_value = cell_t;
    lstm_value.state_active_value = cell_pre_act_t;
    float cell_clip = 0.0;

    lite::arm::math::LstmUnitFunctor<float>::compute(lstm_value,
                                                     frame_size,
                                                     cur_batch_size,
                                                     cell_clip,
                                                     cand_act,
                                                     gate_act,
                                                     cell_act,
                                                     ctx->threads());
    lstm_value.prev_state_value = lstm_value.state_value;

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
