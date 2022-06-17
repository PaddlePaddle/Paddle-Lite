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

#include "lite/kernels/xpu/rnn_compute.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void reset_parameter_vector(const std::vector<Tensor*>& raw_params_vec,
                            const int& num_layers,
                            const int& gate_num,
                            const bool& is_bidirec,
                            std::vector<std::vector<Tensor>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    std::vector<Tensor> tensor_list;
    tensor_list.reserve(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      Tensor tensor_holder;
      tensor_list.emplace_back(tensor_holder);
    }
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      tensor_list[j].ShareDataWith(*raw_params_vec[tensor_idx]);
    }
    params_vec->emplace_back(tensor_list);
  }
}

void runLSTMLayer(xdnn::Context* ctx,
                  int seq_len,
                  int batch_size,
                  int xdim,
                  int hdim,
                  const float* cur_input,
                  float* cur_output,
                  const float* init_h,
                  const float* init_c,
                  float* last_h,
                  float* last_c,
                  int state_offset,
                  const Tensor* sequence_length,
                  const std::vector<Tensor>& vec,
                  bool is_bidirect,
                  int layer_idx,
                  int offset) {
  bool is_reverse = false;
  if (is_bidirect) {
    layer_idx = 2 * layer_idx + offset;
    if (offset > 0) {
      is_reverse = true;
    }
  }
  const int64_t* x_seq_len = nullptr;
  if (sequence_length != nullptr) {
    x_seq_len = sequence_length->data<int64_t>();
  }
  const float* wx = vec[0 + offset * 4].data<float>();
  const float* wh = vec[1 + offset * 4].data<float>();
  const float* bx = vec[2 + offset * 4].data<float>();
  const float* bh = vec[3 + offset * 4].data<float>();

  const float* cur_init_h = init_h + layer_idx * state_offset;
  const float* cur_init_c = init_c + layer_idx * state_offset;
  float* cur_last_h = last_h + (2 * layer_idx + offset) * state_offset;
  float* cur_last_c = last_c + (2 * layer_idx + offset) * state_offset;

  int ret = xdnn::lstm_inference(ctx,
                                 seq_len,
                                 batch_size,
                                 xdim,
                                 hdim,
                                 is_reverse,
                                 cur_input,
                                 cur_init_h,
                                 cur_init_c,
                                 x_seq_len,
                                 wx,
                                 nullptr,
                                 wh,
                                 nullptr,
                                 bx,
                                 bh,
                                 cur_output,
                                 cur_last_h,
                                 cur_last_c);
  CHECK_EQ(ret, 0);
}

void RnnCompute::Run() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<operators::RnnParam>();
  // INPUT
  auto input = param.Input;
  auto pre_state = param.PreState;
  auto weight_list = param.WeightList;
  const Tensor* sequence_length = param.SequenceLength;
  // OUTPUT
  auto output = param.Out;
  auto state = param.State;
  // ATTRIBUTES
  int hidden_size = param.hidden_size;
  int input_size = param.input_size;
  bool is_bidirec = param.is_bidirec;
  bool is_test = param.is_test;
  CHECK(is_test) << "Only support is_test=true but received " << is_test
                 << " in XPU rnn kernel";
  std::string mode = param.mode;
  CHECK(mode == "LSTM") << "Only support mode=LSTM but received " << mode
                        << " in XPU rnn kernel";
  int num_layers = param.num_layers;

  // WeightList
  int gate_num = 4;
  std::vector<std::vector<Tensor>> parameter_lists;
  parameter_lists.reserve(num_layers);
  reset_parameter_vector(
      weight_list, num_layers, gate_num, is_bidirec, &parameter_lists);
  // INPUT and OUTPUT
  int seq_len = input->dims()[0];
  int batch_size = input->dims()[1];
  // int xdim = input_size;
  int hdim = hidden_size;

  const float* input_ptr = input->data<float>();
  if (is_bidirec) {
    output->Resize({seq_len, batch_size, hdim * 2});
  } else {
    output->Resize({seq_len, batch_size, hdim});
  }
  float* output_ptr = output->mutable_data<float>(TARGET(kXPU));

  XPUScratchPadGuard internal_output_1_guard, internal_output_2_guard;
  float* internal_output_1_ptr = nullptr;
  float* internal_output_2_ptr = nullptr;
  if (num_layers >= 2) {
    internal_output_1_guard =
        TargetWrapperXPU::MallocScratchPad(output->numel() * sizeof(float));
    internal_output_1_ptr =
        reinterpret_cast<float*>(internal_output_1_guard->addr_);
  }
  if (num_layers >= 3) {
    internal_output_2_guard =
        TargetWrapperXPU::MallocScratchPad(output->numel() * sizeof(float));
    internal_output_2_ptr =
        reinterpret_cast<float*>(internal_output_2_guard->addr_);
  }
  // PreState and State
  const float* init_h_ptr = pre_state[0]->data<float>();
  const float* init_c_ptr = pre_state[1]->data<float>();
  state[0]->Resize({pre_state[0]->dims()[0], batch_size, hdim * 2});
  float* last_h_ptr = state[0]->mutable_data<float>(TARGET(kXPU));
  state[1]->Resize({pre_state[0]->dims()[0], batch_size, hdim * 2});
  float* last_c_ptr = state[1]->mutable_data<float>(TARGET(kXPU));
  int state_offset = pre_state[0]->dims()[1] * pre_state[0]->dims()[2];

  for (int i = 0; i < num_layers; i++) {
    const float* cur_input_ptr = nullptr;
    int cur_xdim = -1;
    if (i == 0) {
      cur_input_ptr = input_ptr;
      cur_xdim = input_size;
    } else if (i % 2 != 0) {
      cur_input_ptr = internal_output_1_ptr;
      cur_xdim = is_bidirec ? 2 * hdim : hdim;
    } else {
      cur_input_ptr = internal_output_2_ptr;
      cur_xdim = is_bidirec ? 2 * hdim : hdim;
    }
    float* cur_output_ptr = nullptr;
    if (i == num_layers - 1) {
      cur_output_ptr = output_ptr;
    } else if (i % 2 != 0) {
      cur_output_ptr = internal_output_2_ptr;
    } else {
      cur_output_ptr = internal_output_1_ptr;
    }

    if (is_bidirec) {
      std::vector<XPUScratchPadGuard> output_vec(2);
      std::vector<float*> output_ptr_vec(2);
      for (int i = 0; i < 2; ++i) {
        output_vec[i] = TargetWrapperXPU::MallocScratchPad(
            seq_len * batch_size * hdim * sizeof(float));
        output_ptr_vec[i] = reinterpret_cast<float*>(output_vec[i]->addr_);
      }

      runLSTMLayer(ctx.GetRawContext(),
                   seq_len,
                   batch_size,
                   cur_xdim,
                   hdim,
                   cur_input_ptr,
                   output_ptr_vec[0],
                   init_h_ptr,
                   init_c_ptr,
                   last_h_ptr,
                   last_c_ptr,
                   state_offset,
                   sequence_length,
                   parameter_lists[i],
                   is_bidirec,
                   i,
                   0);

      runLSTMLayer(ctx.GetRawContext(),
                   seq_len,
                   batch_size,
                   cur_xdim,
                   hdim,
                   cur_input_ptr,
                   output_ptr_vec[1],
                   init_h_ptr,
                   init_c_ptr,
                   last_h_ptr,
                   last_c_ptr,
                   state_offset,
                   sequence_length,
                   parameter_lists[i],
                   is_bidirec,
                   i,
                   1);
      // concat
      std::vector<const float*> x_list{output_ptr_vec[0], output_ptr_vec[1]};
      std::vector<std::vector<int>> xdims_list{{seq_len, batch_size, hdim},
                                               {seq_len, batch_size, hdim}};

      int r = xdnn::concat<float>(
          ctx.GetRawContext(), x_list, cur_output_ptr, xdims_list, 2);
      CHECK_EQ(r, 0);
    } else {
      runLSTMLayer(ctx.GetRawContext(),
                   seq_len,
                   batch_size,
                   cur_xdim,
                   hdim,
                   cur_input_ptr,
                   cur_output_ptr,
                   init_h_ptr,
                   init_c_ptr,
                   last_h_ptr,
                   last_c_ptr,
                   state_offset,
                   sequence_length,
                   parameter_lists[i],
                   is_bidirec,
                   i,
                   0);
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    rnn, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::RnnCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("WeightList", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PreState", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SequenceLength", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("DropoutState", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Reserve", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("State", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
