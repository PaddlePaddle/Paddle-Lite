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

#include "lite/kernels/x86/rnn_compute.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/host/math/split.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/concat_and_split.h"
#include "lite/backends/x86/math/rnn.h"
#include "lite/kernels/x86/rnn_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

#define RUN_LSTM_LAYER(x, y, z, w) \
  runLSTMLayer(&ctx,               \
               input_temp_holder,  \
               parameter_lists[x], \
               init_h_unbind,      \
               init_c_unbind,      \
               sequence_length,    \
               &last_h_unbind,     \
               &last_c_unbind,     \
               y,                  \
               x,                  \
               &gate_value,        \
               z,                  \
               w)

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

void SwapPoniter(Tensor** a, Tensor** b) {
  Tensor* c = *a;
  *a = *b;
  *b = c;
}

void preprocess(X86Context* ctx,
                const Tensor* input,
                const Tensor& weight,
                const Tensor& bias_ih,
                const Tensor& bias_hh,
                Tensor* cache_input,
                bool is_test) {
  const int& hidden_size = weight.dims()[0];
  int time_step = input->dims()[0];
  int batch = input->dims()[1];

  std::vector<int64_t> cache_input_dim = {time_step, batch, hidden_size};
  DDim gate_dim;
  gate_dim.ConstructFrom(cache_input_dim);
  cache_input->Resize(gate_dim);

  auto* i_data = input->data<float>();
  auto* w_data = weight.data<float>();
  auto* o_data = cache_input->mutable_data<float>();
  auto input_dims = input->dims();
  auto weight_input_dims = weight.dims();
  int m = input_dims[0] * input_dims[1];
  int k = input_dims[2];
  int n = weight_input_dims[0];

  paddle::lite::x86::math::Blas<lite::TargetType::kX86> matmul(*ctx);
  matmul.GEMM<float>(
      false, true, m, n, k, 1.f, i_data, k, w_data, k, 0.f, o_data, n);
  lite::x86::math::fill_bias_fc(o_data, bias_ih.data<float>(), m, n);
  lite::x86::math::fill_bias_fc(o_data, bias_hh.data<float>(), m, n);
}

void cell(X86Context* ctx,
          Tensor* input,
          Tensor* weight_hh,
          Tensor* init_h,
          Tensor* init_c,
          Tensor* last_h,
          Tensor* last_c,
          Tensor* last_c_act,
          Tensor* output,
          const Tensor* bias_hh) {
  auto h_dims = init_h->dims();
  auto weight_input_dims = weight_hh->dims();
  int m = h_dims[0];
  int k = h_dims[1];
  int n = weight_input_dims[0];
  auto i_data = input->data<float>();
  auto w_data = weight_hh->data<float>();
  auto h_data = init_h->data<float>();

  Tensor tmp_gate;
  tmp_gate.Resize(input->dims());
  auto tmp_data = tmp_gate.mutable_data<float>();

  paddle::lite::x86::math::Blas<lite::TargetType::kX86> matmul(*ctx);
  matmul.GEMM<float>(
      false, true, m, n, k, 1.f, h_data, k, w_data, k, 0.f, tmp_data, n);
  for (int i = 0; i < input->dims()[0] * input->dims()[1]; i++) {
    tmp_data[i] += i_data[i];
  }

  Tensor tmp_init_c;
  tmp_init_c.Resize(init_c->dims());
  auto tmp_init_c_data = tmp_init_c.mutable_data<float>();
  for (int i = 0; i < tmp_init_c.dims()[0] * tmp_init_c.dims()[1]; i++) {
    tmp_init_c_data[i] = init_c->data<float>()[i];
  }

  lite::x86::math::LstmMetaValue<float> lstm_value;
  lstm_value.check_ig = nullptr;
  lstm_value.check_fg = nullptr;
  lstm_value.check_og = nullptr;
  lite_api::ActivationType gate_act = lite_api::ActivationType::kSigmoid_v2;
  lite_api::ActivationType cell_act = lite_api::ActivationType::kTanh_v2;
  lite_api::ActivationType cand_act = lite_api::ActivationType::kTanh_v2;

  size_t frame_size = init_h->dims()[1];
  size_t batch_size = init_h->dims()[0];
  Tensor cell_pre_act;
  if (last_c_act == nullptr) {
    cell_pre_act.Resize(init_h->dims());
    cell_pre_act.mutable_data<float>();
    last_c_act = &cell_pre_act;
  }

  lstm_value.prev_state_value = tmp_init_c_data;
  lstm_value.gate_value = tmp_data;
  lstm_value.output_value = output->mutable_data<float>();
  lstm_value.state_value = last_c->mutable_data<float>();
  lstm_value.state_active_value = last_c_act->mutable_data<float>();
  float cell_clip = 0.0;
  lite::x86::math::RnnLstmUnitFunctor<float>::compute(lstm_value,
                                                      frame_size,
                                                      batch_size,
                                                      cell_clip,
                                                      cand_act,
                                                      gate_act,
                                                      cell_act,
                                                      1);
}

void runLSTMLayer(X86Context* ctx,
                  const Tensor* input,
                  std::vector<Tensor> vec,
                  std::vector<Tensor> init_h,
                  std::vector<Tensor> init_c,
                  const Tensor* sequence_length,
                  std::vector<Tensor>* last_h_ptr,
                  std::vector<Tensor>* last_c_ptr,
                  Tensor* output,
                  int layer_idx,
                  Tensor* gate_value,
                  bool is_bidirect,
                  int offset) {
  bool is_reverse = false;
  if (is_bidirect) {
    layer_idx = 2 * layer_idx + offset;
    if (offset > 0) {
      is_reverse = true;
    }
  }

  const int& time_step = input->dims()[0];
  preprocess(ctx,
             input,
             vec[0 + offset * 4],
             vec[2 + offset * 4],
             vec[3 + offset * 4],
             gate_value,
             true);
  std::vector<Tensor> input_tensors, output_tensors;
  std::vector<Tensor *> input_tensors_t, output_tensors_t;
  std::vector<int> stride1, stride2;
  input_tensors.resize(gate_value->dims()[0]);  // time_step
  output_tensors.resize(output->dims()[0]);

  // alloc input
  for (int i = 0; i < gate_value->dims()[0]; i++) {
    stride1.push_back(1);
    int dim1 = gate_value->dims()[1];  // batch
    int dim2 = gate_value->dims()[2];  // hidden
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    input_tensors[i].Resize(dims);
    input_tensors_t.push_back(&input_tensors[i]);
  }

  // alloc output
  for (int i = 0; i < output->dims()[0]; i++) {
    stride2.push_back(1);
    int dim1 = output->dims()[1];
    int dim2 = output->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    output_tensors[i].Resize(dims);
    output_tensors_t.push_back(&output_tensors[i]);
  }

  lite::host::math::split(
      gate_value->data<float>(), input_tensors_t, 0, stride1);
  lite::host::math::split(output->data<float>(), output_tensors_t, 0, stride2);
  auto sd = output->mutable_data<float>();

  if (is_reverse) {
    std::reverse(input_tensors.begin(), input_tensors.end());
  }
  bool has_sequence_length = false;
  /*
    TODO has_sequence_length
  */

  int mask_min_length = time_step;
  if (is_reverse) {
    mask_min_length = mask_min_length - time_step + 1;
  }
  bool has_allocate_mem_c = false;
  bool has_use_last_h_holder = false;
  const int& reverse_flag = is_reverse ? -1 : 1;
  Tensor init_h_temp;
  init_h_temp.CopyDataFrom(init_h[layer_idx]);
  Tensor* init_h_holder = &init_h_temp;
  Tensor* last_h_holder = nullptr;

  if (0 < mask_min_length) {
    last_h_holder = &(output_tensors[0]);
  } else {
    last_h_holder = &(*last_h_ptr)[layer_idx];
    has_use_last_h_holder = true;
  }

  Tensor* init_c_holder = nullptr;
  Tensor* init_c_temp_holder = nullptr;
  Tensor init_c_temp;
  Tensor* last_c_holder = nullptr;
  Tensor last_c_temp;
  last_c_holder = &(*last_c_ptr)[layer_idx];
  init_c_temp_holder = &init_c[layer_idx];

  for (int i = 0; i < time_step; i++) {
    bool in_mask = (reverse_flag * i) >= mask_min_length;
    if (i > 0) {
      if (!has_allocate_mem_c) {
        init_c_temp.Resize(init_h[layer_idx].dims());
        init_c_temp.mutable_data<float>();
        init_c_holder = &init_c_temp;
        has_allocate_mem_c = true;
      }
      SwapPoniter(&init_c_holder, &last_c_holder);
      init_c_temp_holder = init_c_holder;
    }

    // LSTMCELL
    cell(ctx,
         &input_tensors[i],
         &vec[1 + offset * 4],
         init_h_holder,
         init_c_temp_holder,
         last_h_holder,
         last_c_holder,
         nullptr,
         &output_tensors[i],
         &vec[3 + offset * 4]);

    if (in_mask) {
      /*
        TODO in_mask
      */
    }

    // prepare next step
    if (i + 1 < time_step) {
      bool next_step_mask = (reverse_flag * (i + 1)) >= mask_min_length;
      if (next_step_mask) {
        if (!has_use_last_h_holder) {
          init_h_holder = &(*last_h_ptr)[layer_idx];
        }
      } else {
        init_h_holder = &(output_tensors[i + 1]);
      }
      SwapPoniter(&init_h_holder, &last_h_holder);
    }
  }
  if (is_reverse) {
    std::reverse(output_tensors.begin(), output_tensors.end());
  }
  for (int i = 0; i < time_step; i++) {
    int st = output_tensors[i].dims()[0] * output_tensors[i].dims()[1];
    for (int j = 0; j < st; j++) {
      sd[i * st + j] = output_tensors[i].data<float>()[j];
    }
  }

  if (has_sequence_length) {
    if (last_h_holder != &(*last_h_ptr)[layer_idx]) {
      (*last_h_ptr)[layer_idx].CopyDataFrom(*last_h_holder);
    }
  } else {
    (*last_h_ptr)[layer_idx].CopyDataFrom(output_tensors[time_step - 1]);
  }
  if (time_step % 2 == 0) {
    (*last_c_ptr)[layer_idx].CopyDataFrom(*last_c_holder);
  }
}

void RnnCompute::Run() {
  auto& param = this->Param<operators::RnnParam>();
  auto& ctx = this->ctx_->As<X86Context>();
  std::string mode = param.mode;
  auto input = param.Input;
  auto weight_list = param.WeightList;
  auto pre_state = param.PreState;
  auto state = param.State;
  auto output = param.Out;
  bool is_bidirec = param.is_bidirec;
  int num_layers = param.num_layers;
  const Tensor* sequence_length = param.SequenceLength;

  state[0]->mutable_data<float>();
  state[1]->mutable_data<float>();

  // lstmCell begin
  int gate_num = 4;
  std::vector<std::vector<Tensor>> parameter_lists;
  parameter_lists.reserve(num_layers);
  reset_parameter_vector(
      weight_list, num_layers, gate_num, is_bidirec, &parameter_lists);
  Tensor* input_holder;
  Tensor* output_holder = output;
  Tensor temp, gate_value;
  bool has_allocate_mem = false;

  std::vector<Tensor> init_h_unbind, init_c_unbind, last_h_unbind,
      last_c_unbind;
  std::vector<Tensor *> init_h_unbind_t, init_c_unbind_t, last_h_unbind_t,
      last_c_unbind_t;
  init_h_unbind.resize(4);
  init_c_unbind.resize(pre_state[1]->dims()[0]);
  last_h_unbind.resize(state[0]->dims()[0]);
  last_c_unbind.resize(state[1]->dims()[0]);
  std::vector<int> stride;
  for (int i = 0; i < pre_state[0]->dims()[0]; i++) {
    stride.push_back(1);
    int dim1 = pre_state[0]->dims()[1];
    int dim2 = pre_state[0]->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    init_h_unbind[i].Resize(dims);
    init_c_unbind[i].Resize(dims);
    last_h_unbind[i].Resize(dims);
    last_c_unbind[i].Resize(dims);
    init_h_unbind_t.push_back(&init_h_unbind[i]);
    init_c_unbind_t.push_back(&init_c_unbind[i]);
    last_h_unbind_t.push_back(&last_h_unbind[i]);
    last_c_unbind_t.push_back(&last_c_unbind[i]);
  }
  lite::host::math::split(
      pre_state[0]->data<float>(), init_h_unbind_t, 0, stride);
  lite::host::math::split(
      pre_state[1]->data<float>(), init_c_unbind_t, 0, stride);
  lite::host::math::split(state[0]->data<float>(), last_h_unbind_t, 0, stride);
  lite::host::math::split(state[1]->data<float>(), last_c_unbind_t, 0, stride);

  for (int i = 0; i < num_layers; i++) {
    if (i > 0) {
      if (!has_allocate_mem) {
        temp.Resize(output->dims());
        temp.mutable_data<float>();
        input_holder = &temp;
        has_allocate_mem = true;
      }
      SwapPoniter(&output_holder, &input_holder);
    }

    const Tensor* input_temp_holder = input;
    if (i > 0) {
      input_temp_holder = input_holder;
    }

    if (is_bidirec) {
      std::vector<Tensor> output_vec(2);
      int time_step = input->dims()[0];
      int batch_size = input->dims()[1];
      int hidden_size = output->dims()[2];
      for (int i = 0; i < 2; ++i) {
        output_vec[i].Resize({time_step, batch_size, hidden_size / 2});
        output_vec[i].mutable_data<float>();
      }

      RUN_LSTM_LAYER(i, &output_vec[0], true, 0);
      RUN_LSTM_LAYER(i, &output_vec[1], true, 1);

      paddle::lite::x86::math::ConcatFunctor<lite::TargetType::kX86, float>
          concat_x86;
      concat_x86(ctx, output_vec, 2, output);
    } else {
      RUN_LSTM_LAYER(i, output_holder, false, 0);
    }
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    rnn, kX86, kFloat, kNCHW, paddle::lite::kernels::x86::RnnCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("WeightList", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("PreState", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("SequenceLength", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("DropoutState", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Reserve", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("State", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
