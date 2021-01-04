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

void reset_parameter_vector(const std::vector<TensorType>& raw_params_vec,
                            const int& num_layers,
                            const int& gate_num,
                            const bool& is_bidirec,
                            std::vector<std::<Tensor>>* params_vec) {
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

void preprocess(const Tensor* input,
                const Tensor& weight,
                const Tensor& bias_ih,
                const Tensor& bias_hh,
                Tensor* cache_input,
                bool is_test) {
  // crate the temp input for the X * W_ih^T + Bias_ih
  const int& hidden_size = weight.dims()[0];
  int time_step = input->dims()[0];
  int batch = input->dims()[1];
  std::vector<int64_t> cache_input_dim = {time_step, batch, hidden_size};
  auto cache_input_dim = cache_input->dims();
  cache_input_dim.ConstructFrom(cache_input_dim);
  cache_input->Resize(cache_input_dim);
  cache_input->mutable_data<T>();

  auto* i_data = input->data<float>();
  auto* w_data = weight.data();
  auto* o_data = cache_input->mutable_data<float>();

  bool flag_act = false;
  operators::ActivationParam act_param;
  act_param.has_active = false;
  auto input_dims = input->dims();
  auto weight_input_dims = weight.dims();
  auto input_dims = input->dims();
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
                         o_data,
                         n,
                         nullptr,
                         false,
                         act_param,
                         &ctx);

  lite::arm::math::fill_bias_fc(o_data, bias_ih->data<float>(), m, n, flag_act);
  lite::arm::math::fill_bias_fc(o_data, bias_hh->data<float>(), m, n, flag_act);
}

void runLSTMCell(const Tensor* input,
                 vector<Tensor> vec,
                 std::vector<Tensor> init_h,
                 std::vector<Tensor> init_c,
                 const Tensor* sequence_length,
                 std::vector<Tensor>* last_h_ptr,
                 std::vector<Tensor>* last_c_ptr,
                 std::vector<Tensor*> output,
                 int layer_idx,
                 Tensor* gate_value,
                 bool is_bidirect,
                 int offset);
{
  bool is_reverse = false;
  if (is_bidirect) {
    layer_idx = 2 * layer_idx + offset;
    if (offset > 0) {
      is_reverse = true;
    }
  }
  const int& time_step = input->dims()[0];
  preprocess(context,
             input,
             vec[0 + offset * 4],
             vec[2 + offset * 4],
             vec[3 + offset * 4],
             gate_value,
             true);
  // split tensors in axis 0
  auto input_tensors = Unbind(*gate_value);
  auto output_tensors = Unbind(*output);
  if (is_reverse) {
    std::reverse(input_tensors.begin(), input_tensors.end());
    std::reverse(output_tensors.begin(), output_tensors.end());
  }

  /*
  TensorList mask_tensor_list;
  // construct the mask matrix for the mask
  bool has_sequence_length = false;
  if (sequence_length != nullptr) {
    has_sequence_length = true;
  }
  Tensor mask_matrix;
  if (has_sequence_length) {
    DDldim mask_dims = {time_step, input->dims()[1]};
    mask_matrix.Resize(mask_dims);

    create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse,
                          &mask_min_length);
    mask_tensor_list = Unbind(mask_matrix);
  }
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
  const Tensor* init_c_temp_holder = nullptr;
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
    /*
    cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4], init_h_holder,
          init_c_temp_holder, last_h_holder, last_c_holder, nullptr,
          &output_tensors[i], &vec[3 + offset * 4],
          &weight_hh_tmp);
    */
    if (in_mask) {
      std::cout << "in mask" << std::endl;
      // this->postprocess(context, &output_tensors[i], init_h_holder,
      //                  init_c_temp_holder, last_h_holder, last_c_holder,
      //                  mask_tensor_list[i]);
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
  auto& ctx = this->ctx_->As<ARMContext>();
  std::string mode = param.mode;
  auto input = param.Input;
  auto weight_list = param.WeightList;
  auto reserve_data = param.Reserve;
  auto pre_state = param.PreState;
  auto state = param.state;
  auto dropout_state = param.DropoutState auto output = param.Out;
  bool is_bidirec = param.is_bidirec;
  int num_layers = param.num_layers;
  int input_size = param.input_size;
  int hidden_size = param.hidden_size;
  bool is_test = param.is_test;
  float dropout_prob = param.dropout_prob;
  int seed = param.seed;
  const Tensor* sequence_length = param.SequenceLength;

  dropout_state->Resize(output->dims());
  dropout_state->mutable_date<uint8_t>();
  output->mutable_data<float>();
  state[0]->mutable_date<float>();
  state[1]->mutable_date<float>();

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

  /*
   TODO unbind
  */
  auto init_h_unbind = Unbind(*init_h);
  auto last_h_unbind = Unbind(*last_h);
  TensorList init_c_unbind, last_c_unbind;
  if (param.Mode == "LSTM") {
    init_c_unbind = Unbind(*init_c);
    last_c_unbind = Unbind(*last_c);
  }

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

      runLSTMCell(input_temp_holder,
                  parameter_lists[i],
                  init_h_unbind,
                  init_c_unbind,
                  sequence_length,
                  last_h_unbind,
                  last_c_unbind,
                  output_holder,
                  &gate_value,
                  i,
                  true,
                  0);
      runLSTMCell(input_temp_holder,
                  parameter_lists[i],
                  init_h_unbind,
                  init_c_unbind,
                  sequence_length,
                  last_h_unbind,
                  last_c_unbind,
                  output_holder,
                  i,
                  &gate_value,
                  true,
                  1);

      lite::arm::math::concat_func(output_vec, 2, output);

    } else {
      runLSTMCell(input_temp_holder,
                  parameter_lists[i],
                  init_h_unbind,
                  init_c_unbind,
                  sequence_length,
                  last_h_unbind,
                  last_c_unbind,
                  output_holder,
                  i,
                  &gate_value,
                  false,
                  0);
    }
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
