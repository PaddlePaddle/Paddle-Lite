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

#include "lite/kernels/arm/lstm_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/lstm.h"
#include "lite/backends/arm/math/sequence2batch.h"
#include "lite/backends/arm/math/sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T>
void LstmCompute<T>::Run() {
  auto& param = this->Param<operators::LstmParam>();
  auto input = param.Input;
  auto weight = param.Weight;
  auto bias = param.Bias;
  auto hidden_t0 = param.H0;
  auto cell_t0 = param.C0;
  auto batch_gate = param.BatchGate;
  auto hidden_out = param.Hidden;
  auto cell_out = param.Cell;
  auto batch_cell_pre_act = param.BatchCellPreAct;

  batch_gate->template mutable_data<T>();
  hidden_out->template mutable_data<T>();
  cell_out->template mutable_data<T>();

  bool is_reverse = param.is_reverse;
  lite::arm::math::LoDTensor2BatchFunctor<T> to_batch;
  to_batch(*input, batch_gate, true, is_reverse);

  auto in_dims = input->dims();
  int frame_size = static_cast<int>(in_dims[1] / 4);
  DDimLite dims(std::vector<int64_t>{in_dims[0], frame_size});

  if (bias) {
    // checkpoint1
    lite::arm::math::add_bias_rowwise(batch_gate, bias, 0, 4 * frame_size);
  }

  lite::arm::math::LstmMetaValue<T> lstm_value;
  if (bias && param.use_peepholes) {
    T* bias_data = const_cast<T*>(bias->template data<T>());
    // the code style in LstmMetaValue will be updated later.
    lstm_value.check_ig = bias_data + 4 * frame_size;
    lstm_value.check_fg = lstm_value.check_ig + frame_size;
    lstm_value.check_og = lstm_value.check_fg + frame_size;
  } else {
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;
  }
  lstm_value.prev_state_value = nullptr;
  Tensor ordered_c0;

  std::vector<uint64_t> order(batch_gate->lod()[2]);

  if (cell_t0) {
    // Since the batch computing for LSTM reorders the input sequence
    // according to their length. The initialized cell state also needs
    // to reorder.
    lite::arm::math::ReorderInitState<T>(*cell_t0, order, &ordered_c0, true);
    lstm_value.prev_state_value = ordered_c0.mutable_data<T>();
  }
  // Use the local variable as here.
  Tensor batch_hidden, batch_cell;
  batch_hidden.Resize(dims);
  batch_cell.Resize(dims);
  batch_cell_pre_act->Resize(dims);
  batch_hidden.mutable_data<T>();
  batch_cell.mutable_data<T>();
  batch_cell_pre_act->template mutable_data<T>();

  auto batch_starts = batch_gate->lod()[0];
  size_t num_batch = batch_starts.size() - 1;

  std::string gate_act = param.gate_activation;
  std::string cell_act = param.cell_activation;
  std::string cand_act = param.candidate_activation;

  int matrix_width = batch_gate->numel() / in_dims[0];
  auto& ctx = this->ctx_->template As<ARMContext>();
  for (size_t n = 0; n < num_batch; n++) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    auto gate_t = lite::arm::math::row_offset(*batch_gate, bstart);
    auto out_t = lite::arm::math::row_offset(batch_hidden, bstart);
    auto cell_t = lite::arm::math::row_offset(batch_cell, bstart);
    auto cell_pre_act_t =
        lite::arm::math::row_offset(*batch_cell_pre_act, bstart);

    int cur_batch_size = bend - bstart;
    operators::ActivationParam act_param;
    act_param.has_active = false;

    if (n > 0) {
      int pre_h_start = static_cast<int>(batch_starts[n - 1]);
      int pre_h_end = pre_h_start + cur_batch_size;

      auto pre_hidden_t =
          lite::arm::math::row_offset(batch_hidden, pre_h_start);
      int M = pre_h_end - pre_h_start;
      int N = matrix_width;
      int K = frame_size;

      lite::arm::math::sgemm(false,
                             false,
                             M,
                             N,
                             K,
                             1,
                             pre_hidden_t,
                             K,
                             weight->template data<T>(),
                             N,
                             1,
                             gate_t,
                             N,
                             nullptr,
                             false,
                             act_param,
                             &ctx);
    } else if (hidden_t0) {
      // If n == 0 and there is no initialized hidden state, that is to say
      // the H0 is zeros, the calculation W_h * H0 will be skiped.
      // If n == 0 and there is initialized hidden state, calculate W_h * H0.
      // Since the batch computing for LSTM reorders the input sequence
      // according to their length. The initialized hidden state also needs
      // to reorder.
      Tensor ordered_h0;
      lite::arm::math::ReorderInitState<T>(
          *hidden_t0, order, &ordered_h0, true);
      int M = ordered_h0.dims()[0];
      int N = matrix_width;
      int K = frame_size;
      lite::arm::math::sgemm(false,
                             false,
                             M,
                             N,
                             K,
                             1,
                             ordered_h0.data<T>(),
                             K,
                             weight->template data<T>(),
                             N,
                             1,
                             gate_t,
                             N,
                             nullptr,
                             false,
                             act_param,
                             &ctx);
    }

    lstm_value.gate_value = gate_t;
    lstm_value.output_value = out_t;
    lstm_value.state_value = cell_t;
    lstm_value.state_active_value = cell_pre_act_t;
    T cell_clip = 0.0;
    // checkpoint
    lite::arm::math::LstmUnitFunctor<T>::compute(lstm_value,
                                                 frame_size,
                                                 cur_batch_size,
                                                 cell_clip,
                                                 cand_act,
                                                 gate_act,
                                                 cell_act,
                                                 ctx.threads());
    lstm_value.prev_state_value = lstm_value.state_value;
  }

  lite::arm::math::Batch2LoDTensorFunctor<T> to_seq;
  auto* lod_hidden = batch_hidden.mutable_lod();
  *lod_hidden = batch_gate->lod();
  to_seq(batch_hidden, hidden_out);
  auto* lod_cell = batch_cell.mutable_lod();
  *lod_cell = batch_gate->lod();
  to_seq(batch_cell, cell_out);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lstm,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LstmCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("C0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Cell", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchCellPreAct", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
