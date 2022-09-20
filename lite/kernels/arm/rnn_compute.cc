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
#include "lite/backends/arm/math/concat.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/gru.h"
#include "lite/backends/arm/math/lstm.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/backends/host/math/split.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

// layer, output_tensor, is_bidirection, offset
#define RUN_RNN_LAYER(x, y, z, w) \
  RunRnnLayer(&ctx,               \
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
              w,                  \
              mode)

#ifdef ENABLE_ARM_FP16
// layer, output_tensor, is_bidirection, offset
#define RUN_RNN_LAYER_FP16(x, y, z, w) \
  RunRnnLayer_fp16(&ctx,               \
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
                   w,                  \
                   mode)
#endif

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

/******************************************************
input:
    ctx:context,
    input:(3D)time_step, batch, input_size,
    weight:(2D)hidden_size, input_size,
    bias_ih,
    bias_hh,
    mode:LSTM, GRU
output:
    cache_input:(3D)time_step, batch, hidden_size
*******************************************************/
static void preprocess(ARMContext* ctx,
                       const Tensor* input,
                       const Tensor& weight,
                       const Tensor& bias_ih,
                       const Tensor& bias_hh,
                       std::string mode,
                       Tensor* cache_input) {
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
  operators::ActivationParam act_param;
  act_param.has_active = false;
  auto input_dims = input->dims();
  auto weight_input_dims = weight.dims();
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
                         k,
                         0.f,
                         o_data,
                         n,
                         nullptr,
                         false,
                         act_param,
                         ctx);
  lite::arm::math::fill_bias_fc(o_data, bias_ih.data<float>(), m, n, nullptr);

  if ("GRU" == mode) {
    Tensor bias_tmp_hh;
    bias_tmp_hh.Resize(bias_hh.dims());
    auto bias_ptr = bias_tmp_hh.mutable_data<float>();
    auto bias_src = bias_hh.data<float>();
    int bias_offt = bias_hh.numel() / 3 * 2;
    std::memcpy(bias_ptr, bias_src, bias_offt * sizeof(float));
    std::memset(
        bias_ptr + bias_offt, 0, (bias_hh.numel() - bias_offt) * sizeof(float));
    lite::arm::math::fill_bias_fc(
        o_data, bias_tmp_hh.data<float>(), m, n, nullptr);
  } else {
    lite::arm::math::fill_bias_fc(o_data, bias_hh.data<float>(), m, n, nullptr);
  }
}

#ifdef ENABLE_ARM_FP16
static void preprocess_fp16(ARMContext* ctx,
                            const Tensor* input,
                            const Tensor& weight,
                            const Tensor& bias_ih,
                            const Tensor& bias_hh,
                            std::string mode,
                            Tensor* cache_input) {
  const int& hidden_size = weight.dims()[0];
  int time_step = input->dims()[0];
  int batch = input->dims()[1];

  std::vector<int64_t> cache_input_dim = {time_step, batch, hidden_size};
  DDim gate_dim;
  gate_dim.ConstructFrom(cache_input_dim);
  cache_input->Resize(gate_dim);

  auto* i_data = input->data<float16_t>();
  auto* w_data = weight.data<float16_t>();
  auto* o_data = cache_input->mutable_data<float16_t>();
  operators::ActivationParam act_param;
  act_param.has_active = false;
  auto input_dims = input->dims();
  auto weight_input_dims = weight.dims();
  int m = input_dims[0] * input_dims[1];
  int k = input_dims[2];
  int n = weight_input_dims[0];

  lite::arm::math::fp16::sgemm_fp16(false,
                                    true,
                                    m,
                                    n,
                                    k,
                                    1.f,
                                    i_data,
                                    k,
                                    w_data,
                                    k,
                                    0.f,
                                    o_data,
                                    n,
                                    nullptr,
                                    false,
                                    act_param,
                                    ctx);
  lite::arm::math::fp16::fill_bias_fc(
      o_data, bias_ih.data<float16_t>(), m, n, nullptr);

  if ("GRU" == mode) {
    Tensor bias_tmp_hh;
    bias_tmp_hh.Resize(bias_hh.dims());
    auto bias_ptr = bias_tmp_hh.mutable_data<float16_t>();
    auto bias_src = bias_hh.data<float16_t>();
    int bias_offt = bias_hh.numel() / 3 * 2;
    std::memcpy(bias_ptr, bias_src, bias_offt * sizeof(float16_t));
    std::memset(bias_ptr + bias_offt,
                0,
                (bias_hh.numel() - bias_offt) * sizeof(float16_t));
    lite::arm::math::fp16::fill_bias_fc(
        o_data, bias_tmp_hh.data<float16_t>(), m, n, nullptr);
  } else {
    lite::arm::math::fp16::fill_bias_fc(
        o_data, bias_hh.data<float16_t>(), m, n, nullptr);
  }
}
#endif

/******************************************************
input:
    ctx:context,
    init_h:(2D),
    init_c:(2D),
    mask_tensor:(1D)input->dims()[1],
    mode:LSTM, GRU
output:
    output:(2D)output->dims()[1], output->dims()[2],
    last_h:(2D),
    last_c:(2D)
*******************************************************/
static void postprocess(ARMContext* ctx,
                        Tensor* output,
                        const Tensor* init_h,
                        const Tensor* init_c,
                        Tensor* last_h,
                        Tensor* last_c,
                        const Tensor& mask_tensor,
                        std::string mode) {
  Tensor mask_broadcast_1;
  mask_broadcast_1.Resize(mask_tensor.dims());
  auto mask_ptr_1 = mask_broadcast_1.mutable_data<float>();
  auto mask_ptr = mask_tensor.data<float>();
  auto out_ptr = output->mutable_data<float>();
  auto cur_h_ptr = last_h->mutable_data<float>();
  auto pre_h_ptr = init_h->data<float>();
  int offset = 0;

  // out = out * mask_broadcast
  // curr_h = out * mask_broadcast + pre_h * (1 - mask_broadcast);
  for (int i = 0; i < output->dims()[0]; i++) {
    mask_ptr_1[i] = 1 - mask_ptr[i];
    for (int j = 0; j < output->dims()[1]; j++) {
      offset = i * output->dims()[1] + j;
      out_ptr[offset] *= mask_ptr[i];
      cur_h_ptr[offset] = out_ptr[offset] + pre_h_ptr[offset] * mask_ptr_1[i];
    }
  }
  if ("LSTM" == mode) {
    auto pre_c_ptr = init_c->data<float>();
    auto cur_c_ptr = last_c->mutable_data<float>();

    // curr_c = curr_c * mask_broadcast + pre_c * (1 - mask_broadcast);
    for (int i = 0; i < output->dims()[0]; i++) {
      for (int j = 0; j < output->dims()[1]; j++) {
        offset = i * output->dims()[1] + j;
        cur_c_ptr[offset] =
            cur_c_ptr[offset] * mask_ptr[i] + pre_c_ptr[offset] * mask_ptr_1[i];
      }
    }
  }
}

#ifdef ENABLE_ARM_FP16
static void postprocess_fp16(ARMContext* ctx,
                             Tensor* output,
                             const Tensor* init_h,
                             const Tensor* init_c,
                             Tensor* last_h,
                             Tensor* last_c,
                             const Tensor& mask_tensor,
                             std::string mode) {
  Tensor mask_broadcast_1;
  mask_broadcast_1.Resize(mask_tensor.dims());
  auto mask_ptr_1 = mask_broadcast_1.mutable_data<float16_t>();
  auto mask_ptr = mask_tensor.data<float16_t>();
  auto out_ptr = output->mutable_data<float16_t>();
  auto cur_h_ptr = last_h->mutable_data<float16_t>();
  auto pre_h_ptr = init_h->data<float16_t>();
  int offset = 0;

  // out = out * mask_broadcast
  // curr_h = out * mask_broadcast + pre_h * (1 - mask_broadcast);
  for (int i = 0; i < output->dims()[0]; i++) {
    mask_ptr_1[i] = 1 - mask_ptr[i];
    for (int j = 0; j < output->dims()[1]; j++) {
      offset = i * output->dims()[1] + j;
      out_ptr[offset] *= mask_ptr[i];
      cur_h_ptr[offset] = out_ptr[offset] + pre_h_ptr[offset] * mask_ptr_1[i];
    }
  }
  if ("LSTM" == mode) {
    auto pre_c_ptr = init_c->data<float16_t>();
    auto cur_c_ptr = last_c->mutable_data<float16_t>();

    // curr_c = curr_c * mask_broadcast + pre_c * (1 - mask_broadcast);
    for (int i = 0; i < output->dims()[0]; i++) {
      for (int j = 0; j < output->dims()[1]; j++) {
        offset = i * output->dims()[1] + j;
        cur_c_ptr[offset] =
            cur_c_ptr[offset] * mask_ptr[i] + pre_c_ptr[offset] * mask_ptr_1[i];
      }
    }
  }
}
#endif

static DDim get_stride(const DDim& ddim) {
  DDim strides = ddim;
  strides[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i + 1];
  }
  return strides;
}

template <typename T>
static void TransposeNormal(const Tensor& in,
                            Tensor* out,
                            const std::vector<int>& axis) {
  const int rank = axis.size();
  auto in_stride = get_stride(in.dims());
  auto out_stride = get_stride(out->dims());
  const T* in_ptr = in.data<T>();
  T* out_ptr = out->mutable_data<T>();

  auto transpose_helper = [&](int64_t beg, int64_t end) {
    for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
      int64_t in_idx = 0;
      int64_t tmp_idx = out_idx;
      // calculate the input index
      for (int i = 0; i < rank; ++i) {
        const int64_t coordinate = tmp_idx / out_stride[i];
        tmp_idx -= coordinate * out_stride[i];
        in_idx += coordinate * in_stride[axis[i]];
      }
      out_ptr[out_idx] = in_ptr[in_idx];
    }
  };
  transpose_helper(0, out->numel());
}

/******************************************************
input:
    sequence_length,
    is_reverse
output:
    mask_matrix,
    min_seq_len
******************************************************/
static void create_mask_matrix(const Tensor* sequence_length,
                               Tensor* mask_matrix,
                               const bool& is_reverse,
                               int* min_seq_len) {
  // Tensor to vector<int>
  std::vector<int> seq_len_vec;
  seq_len_vec.resize(sequence_length->numel());
  std::memcpy(&seq_len_vec[0],
              sequence_length->data<int>(),
              sequence_length->numel() * sizeof(int));

  const int& table_width = mask_matrix->dims()[0];
  Tensor temp;
  DDimLite dims(
      std::vector<int64_t>{mask_matrix->dims()[1], mask_matrix->dims()[0]});
  temp.Resize(dims);
  float* data_temp = temp.mutable_data<float>();
  std::fill(data_temp, data_temp + mask_matrix->numel(), 1.f);
  *min_seq_len = table_width;
  for (unsigned int i = 0; i < seq_len_vec.size(); i++) {
    // reset the mask matrix
    *min_seq_len = std::min(seq_len_vec[i], *min_seq_len);
    if (seq_len_vec[i] == table_width) {
      continue;
    }
    if (is_reverse) {
      std::fill(data_temp + i * table_width,
                data_temp + (i + 1) * table_width - seq_len_vec[i],
                0.f);
    } else {
      std::fill(data_temp + i * table_width + seq_len_vec[i],
                data_temp + (i + 1) * table_width,
                0.f);
    }
  }

  mask_matrix->mutable_data<float>();
  std::vector<int> trans_vec;
  trans_vec.emplace_back(1);
  trans_vec.emplace_back(0);
  TransposeNormal<float>(temp, mask_matrix, trans_vec);
}

#ifdef ENABLE_ARM_FP16
static void create_mask_matrix_fp16(const Tensor* sequence_length,
                                    Tensor* mask_matrix,
                                    const bool& is_reverse,
                                    int* min_seq_len) {
  // Tensor to vector<int>
  std::vector<int> seq_len_vec;
  seq_len_vec.resize(sequence_length->numel());
  std::memcpy(&seq_len_vec[0],
              sequence_length->data<int>(),
              sequence_length->numel() * sizeof(int));

  const int& table_width = mask_matrix->dims()[0];
  Tensor temp;
  DDimLite dims(
      std::vector<int64_t>{mask_matrix->dims()[1], mask_matrix->dims()[0]});
  temp.Resize(dims);
  float16_t* data_temp = temp.mutable_data<float16_t>();
  std::fill(data_temp, data_temp + mask_matrix->numel(), 1.f);
  *min_seq_len = table_width;
  for (unsigned int i = 0; i < seq_len_vec.size(); i++) {
    // reset the mask matrix
    *min_seq_len = std::min(seq_len_vec[i], *min_seq_len);
    if (seq_len_vec[i] == table_width) {
      continue;
    }
    if (is_reverse) {
      std::fill(data_temp + i * table_width,
                data_temp + (i + 1) * table_width - seq_len_vec[i],
                0.f);
    } else {
      std::fill(data_temp + i * table_width + seq_len_vec[i],
                data_temp + (i + 1) * table_width,
                0.f);
    }
  }

  mask_matrix->mutable_data<float16_t>();
  std::vector<int> trans_vec;
  trans_vec.emplace_back(1);
  trans_vec.emplace_back(0);
  TransposeNormal<float16_t>(temp, mask_matrix, trans_vec);
}
#endif

static void lstm_cell(ARMContext* ctx,
                      Tensor* input,
                      Tensor* weight_hh,
                      Tensor* init_h,
                      Tensor* init_c,
                      Tensor* last_h,
                      Tensor* last_c,
                      Tensor* last_c_act,
                      Tensor* output,
                      const Tensor* bias_hh) {
  operators::ActivationParam act_param;
  act_param.has_active = false;
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
  lite::arm::math::sgemm(false,
                         true,
                         m,
                         n,
                         k,
                         1.f,
                         h_data,
                         k,
                         w_data,
                         k,
                         0.f,
                         tmp_data,
                         n,
                         nullptr,
                         false,
                         act_param,
                         ctx);
  for (int i = 0; i < input->dims()[0] * input->dims()[1]; i++) {
    tmp_data[i] += i_data[i];
  }

  Tensor tmp_init_c;
  tmp_init_c.Resize(init_c->dims());
  auto tmp_init_c_data = tmp_init_c.mutable_data<float>();
  for (int i = 0; i < tmp_init_c.dims()[0] * tmp_init_c.dims()[1]; i++) {
    tmp_init_c_data[i] = init_c->data<float>()[i];
  }

  lite::arm::math::LstmMetaValue<float> lstm_value;
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
  lite::arm::math::RnnLstmUnitFunctor<float>::compute(lstm_value,
                                                      frame_size,
                                                      batch_size,
                                                      cell_clip,
                                                      cand_act,
                                                      gate_act,
                                                      cell_act,
                                                      ctx->threads());
}

#ifdef ENABLE_ARM_FP16
static void lstm_cell_fp16(ARMContext* ctx,
                           Tensor* input,
                           Tensor* weight_hh,
                           Tensor* init_h,
                           Tensor* init_c,
                           Tensor* last_h,
                           Tensor* last_c,
                           Tensor* last_c_act,
                           Tensor* output,
                           const Tensor* bias_hh) {
  operators::ActivationParam act_param;
  act_param.has_active = false;
  auto h_dims = init_h->dims();
  auto weight_input_dims = weight_hh->dims();
  int m = h_dims[0];
  int k = h_dims[1];
  int n = weight_input_dims[0];
  auto i_data = input->data<float16_t>();
  auto w_data = weight_hh->data<float16_t>();
  auto h_data = init_h->data<float16_t>();

  Tensor tmp_gate;
  tmp_gate.Resize(input->dims());
  auto tmp_data = tmp_gate.mutable_data<float16_t>();
  lite::arm::math::fp16::sgemm_fp16(false,
                                    true,
                                    m,
                                    n,
                                    k,
                                    1.f,
                                    h_data,
                                    k,
                                    w_data,
                                    k,
                                    0.f,
                                    tmp_data,
                                    n,
                                    nullptr,
                                    false,
                                    act_param,
                                    ctx);
  for (int i = 0; i < input->dims()[0] * input->dims()[1]; i++) {
    tmp_data[i] += i_data[i];
  }

  Tensor tmp_init_c;
  tmp_init_c.Resize(init_c->dims());
  auto tmp_init_c_data = tmp_init_c.mutable_data<float16_t>();
  for (int i = 0; i < tmp_init_c.dims()[0] * tmp_init_c.dims()[1]; i++) {
    tmp_init_c_data[i] = init_c->data<float16_t>()[i];
  }

  lite::arm::math::LstmMetaValue<float16_t> lstm_value;
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
    cell_pre_act.mutable_data<float16_t>();
    last_c_act = &cell_pre_act;
  }

  lstm_value.prev_state_value = tmp_init_c_data;
  lstm_value.gate_value = tmp_data;
  lstm_value.output_value = output->mutable_data<float16_t>();
  lstm_value.state_value = last_c->mutable_data<float16_t>();
  lstm_value.state_active_value = last_c_act->mutable_data<float16_t>();
  float16_t cell_clip = 0.0;
  lite::arm::math::RnnLstmUnitFunctorFP16::compute(lstm_value,
                                                   frame_size,
                                                   batch_size,
                                                   cell_clip,
                                                   cand_act,
                                                   gate_act,
                                                   cell_act,
                                                   ctx->threads());
}
#endif

static void gru_cell(ARMContext* ctx,
                     Tensor* input,
                     Tensor* weight_hh,
                     Tensor* init_h,
                     Tensor* init_c,
                     Tensor* last_h,
                     Tensor* last_c,
                     Tensor* last_c_act,
                     Tensor* output,
                     const Tensor* bias_hh,
                     Tensor* weight_hh_gru) {
  operators::ActivationParam act_param;
  act_param.has_active = false;
  auto h_dims = init_h->dims();
  auto weight_gru_dims = weight_hh_gru->dims();
  int m = h_dims[0];
  int k = h_dims[1];
  int n = weight_gru_dims[0];
  auto i_data = input->data<float>();
  auto w_gru = weight_hh_gru->data<float>();
  auto h_data = init_h->data<float>();

  Tensor tmp_gate;
  tmp_gate.Resize(input->dims());
  auto tmp_data = tmp_gate.mutable_data<float>();
  lite::arm::math::sgemm(false,
                         true,
                         m,
                         n,
                         k,
                         1.f,
                         h_data,
                         k,
                         w_gru,
                         k,
                         0.f,
                         tmp_data,
                         n,
                         nullptr,
                         false,
                         act_param,
                         ctx);
  for (int i = 0; i < input->dims()[0] * input->dims()[1]; i++) {
    tmp_data[i] += i_data[i];
  }

  size_t frame_size = init_h->dims()[1];
  size_t batch_size = init_h->dims()[0];

  lite::arm::math::RNNGRUValue<float> gru_value;
  gru_value.gate_weight = weight_hh->data<float>();
  gru_value.state_weight =
      weight_hh->data<float>() + 2 * frame_size * frame_size;
  gru_value.reset_bias = bias_hh->data<float>() + 2 * frame_size;

  gru_value.gate_value = tmp_data;
  gru_value.reset_output_value = last_c->mutable_data<float>();
  gru_value.output_value = output->mutable_data<float>();
  gru_value.prev_out_value = init_h->data<float>();

  lite_api::ActivationType gate_act = lite_api::ActivationType::kSigmoid_v2;
  lite_api::ActivationType cand_act = lite_api::ActivationType::kTanh_v2;

  lite::arm::math::RnnGruUnitFunctorV2<float>::compute(
      ctx, gru_value, frame_size, batch_size, cand_act, gate_act);
}

#ifdef ENABLE_ARM_FP16
static void gru_cell_fp16(ARMContext* ctx,
                          Tensor* input,
                          Tensor* weight_hh,
                          Tensor* init_h,
                          Tensor* init_c,
                          Tensor* last_h,
                          Tensor* last_c,
                          Tensor* last_c_act,
                          Tensor* output,
                          const Tensor* bias_hh,
                          Tensor* weight_hh_gru) {
  operators::ActivationParam act_param;
  act_param.has_active = false;
  auto h_dims = init_h->dims();
  auto weight_gru_dims = weight_hh_gru->dims();
  int m = h_dims[0];
  int k = h_dims[1];
  int n = weight_gru_dims[0];
  auto i_data = input->data<float16_t>();
  auto w_gru = weight_hh_gru->data<float16_t>();
  auto h_data = init_h->data<float16_t>();

  Tensor tmp_gate;
  tmp_gate.Resize(input->dims());
  auto tmp_data = tmp_gate.mutable_data<float16_t>();
  lite::arm::math::fp16::sgemm_fp16(false,
                                    true,
                                    m,
                                    n,
                                    k,
                                    1.f,
                                    h_data,
                                    k,
                                    w_gru,
                                    k,
                                    0.f,
                                    tmp_data,
                                    n,
                                    nullptr,
                                    false,
                                    act_param,
                                    ctx);
  for (int i = 0; i < input->dims()[0] * input->dims()[1]; i++) {
    tmp_data[i] += i_data[i];
  }

  size_t frame_size = init_h->dims()[1];
  size_t batch_size = init_h->dims()[0];

  lite::arm::math::RNNGRUValue<float16_t> gru_value;
  gru_value.gate_weight = weight_hh->data<float16_t>();
  gru_value.state_weight =
      weight_hh->data<float16_t>() + 2 * frame_size * frame_size;
  gru_value.reset_bias = bias_hh->data<float16_t>() + 2 * frame_size;

  gru_value.gate_value = tmp_data;
  gru_value.reset_output_value = last_c->mutable_data<float16_t>();
  gru_value.output_value = output->mutable_data<float16_t>();
  gru_value.prev_out_value = init_h->data<float16_t>();

  lite_api::ActivationType gate_act = lite_api::ActivationType::kSigmoid_v2;
  lite_api::ActivationType cand_act = lite_api::ActivationType::kTanh_v2;

  lite::arm::math::RnnGruUnitFunctorV2<float16_t>::compute(
      ctx, gru_value, frame_size, batch_size, cand_act, gate_act);
}
#endif

static void RunRnnLayer(ARMContext* ctx,
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
                        int offset,
                        std::string mode) {
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
             mode,
             gate_value);
  std::vector<Tensor> input_tensors, output_tensors;
  std::vector<Tensor *> input_tensors_t, output_tensors_t;
  std::vector<int> stride1, stride2;
  input_tensors.resize(gate_value->dims()[0]);
  output_tensors.resize(output->dims()[0]);

  // unbind
  for (int i = 0; i < gate_value->dims()[0]; i++) {
    stride1.push_back(1);
    int dim1 = gate_value->dims()[1];
    int dim2 = gate_value->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    input_tensors[i].Resize(dims);
    input_tensors_t.push_back(&input_tensors[i]);
  }
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
    // don't need to reverse input_tensors_t becauese of unuseful
    std::reverse(input_tensors.begin(), input_tensors.end());
  }
  bool has_sequence_length = false;
  if (sequence_length != nullptr) {
    has_sequence_length = true;
  }
  // unbind
  Tensor mask_matrix;
  std::vector<Tensor> mask_vec(time_step);
  int mask_min_length = time_step;

  /*
   to be verifying!
  */
  if (has_sequence_length) {
    mask_matrix.Resize(DDimLite({time_step, input->dims()[1]}));
    create_mask_matrix(
        sequence_length, &mask_matrix, is_reverse, &mask_min_length);
    auto mask_matrix_ptr = mask_matrix.data<float>();
    for (int i = 0; i < time_step; i++) {
      DDimLite ddims(std::vector<int64_t>{input->dims()[1]});
      mask_vec[i].Resize(ddims);
      auto tmp_ptr = mask_vec[i].mutable_data<float>();
      for (int j = 0; j < input->dims()[1]; j++) {
        tmp_ptr[j] = mask_matrix_ptr[i * input->dims()[1] + j];
      }
    }
  }

  if (is_reverse) {
    mask_min_length = mask_min_length - time_step + 1;
  }

  bool has_allocate_mem_c = false;
  bool has_use_last_h_holder = false;
  const int& reverse_flag = is_reverse ? -1 : 1;

  // define the init_h holder for the swap
  Tensor init_h_temp;
  init_h_temp.Resize(init_h[layer_idx].dims());
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

  if ("LSTM" == mode) {
    last_c_holder = &(*last_c_ptr)[layer_idx];
    init_c_temp_holder = &init_c[layer_idx];
  } else if ("GRU" == mode) {
    // for reset output value
    last_c_temp.Resize(init_h[layer_idx].dims());
    last_c_temp.mutable_data<float>();
    last_c_holder = &last_c_temp;
  }

  Tensor weight_hh_tmp;  // for gru
  std::vector<Tensor> weight_hh_tmp_ubind;
  std::vector<Tensor*> weight_hh_tmp_ubind_t;
  std::vector<int> stride_w;
  if ("GRU" == mode) {
    weight_hh_tmp.Resize(vec[1 + offset * 4].dims());
    weight_hh_tmp.mutable_data<float>();
    weight_hh_tmp.CopyDataFrom(vec[1 + offset * 4]);
    int size = weight_hh_tmp.numel() / 3;
    std::memset(weight_hh_tmp.mutable_data<float>() + size * 2,
                0,
                size * sizeof(float));
  }

  for (int i = 0; i < time_step; i++) {
    bool in_mask = (reverse_flag * i) >= mask_min_length;
    if (i > 0) {
      if (!has_allocate_mem_c) {
        if (("LSTM" == mode) || ("GRU" == mode)) {
          init_c_temp.Resize(init_h[layer_idx].dims());
          init_c_temp.mutable_data<float>();
          init_c_holder = &init_c_temp;
        }
        has_allocate_mem_c = true;
      }
      SwapPoniter(&init_c_holder, &last_c_holder);
      init_c_temp_holder = init_c_holder;
    }

    if ("LSTM" == mode) {
      lstm_cell(ctx,
                &input_tensors[i],
                &vec[1 + offset * 4],
                init_h_holder,
                init_c_temp_holder,
                last_h_holder,
                last_c_holder,
                nullptr,
                &output_tensors[i],
                &vec[3 + offset * 4]);
    } else if ("GRU" == mode) {
      gru_cell(ctx,
               &input_tensors[i],
               &vec[1 + offset * 4],
               init_h_holder,
               init_c_temp_holder,
               last_h_holder,
               last_c_holder,
               nullptr,
               &output_tensors[i],
               &vec[3 + offset * 4],
               &weight_hh_tmp);
    }

    /*
     to be verifying!
    */
    if (in_mask) {
      postprocess(ctx,
                  &output_tensors[i],
                  init_h_holder,
                  init_c_temp_holder,
                  last_h_holder,
                  last_c_holder,
                  mask_vec[i],
                  mode);
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
  if ((0 == (time_step % 2)) && ("LSTM" == mode)) {
    (*last_c_ptr)[layer_idx].CopyDataFrom(*last_c_holder);
  }
}

#ifdef ENABLE_ARM_FP16
static void RunRnnLayer_fp16(ARMContext* ctx,
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
                             int offset,
                             std::string mode) {
  bool is_reverse = false;
  if (is_bidirect) {
    layer_idx = 2 * layer_idx + offset;
    if (offset > 0) {
      is_reverse = true;
    }
  }
  const int& time_step = input->dims()[0];
  preprocess_fp16(ctx,
                  input,
                  vec[0 + offset * 4],
                  vec[2 + offset * 4],
                  vec[3 + offset * 4],
                  mode,
                  gate_value);
  std::vector<Tensor> input_tensors, output_tensors;
  std::vector<Tensor *> input_tensors_t, output_tensors_t;
  std::vector<int> stride1, stride2;
  input_tensors.resize(gate_value->dims()[0]);
  output_tensors.resize(output->dims()[0]);

  // unbind
  for (int i = 0; i < gate_value->dims()[0]; i++) {
    stride1.push_back(1);
    int dim1 = gate_value->dims()[1];
    int dim2 = gate_value->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    input_tensors[i].Resize(dims);
    input_tensors_t.push_back(&input_tensors[i]);
  }
  for (int i = 0; i < output->dims()[0]; i++) {
    stride2.push_back(1);
    int dim1 = output->dims()[1];
    int dim2 = output->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    output_tensors[i].Resize(dims);
    output_tensors_t.push_back(&output_tensors[i]);
  }
  lite::host::math::split(
      gate_value->data<float16_t>(), input_tensors_t, 0, stride1);
  lite::host::math::split(
      output->data<float16_t>(), output_tensors_t, 0, stride2);
  auto sd = output->mutable_data<float16_t>();
  if (is_reverse) {
    // don't need to reverse input_tensors_t becauese of unuseful
    std::reverse(input_tensors.begin(), input_tensors.end());
  }
  bool has_sequence_length = false;
  if (sequence_length != nullptr) {
    has_sequence_length = true;
  }
  // unbind
  Tensor mask_matrix;
  std::vector<Tensor> mask_vec(time_step);
  int mask_min_length = time_step;

  /*
   to be verifying!
  */
  if (has_sequence_length) {
    mask_matrix.Resize(DDimLite({time_step, input->dims()[1]}));
    create_mask_matrix_fp16(
        sequence_length, &mask_matrix, is_reverse, &mask_min_length);
    auto mask_matrix_ptr = mask_matrix.data<float16_t>();
    for (int i = 0; i < time_step; i++) {
      DDimLite ddims(std::vector<int64_t>{input->dims()[1]});
      mask_vec[i].Resize(ddims);
      auto tmp_ptr = mask_vec[i].mutable_data<float16_t>();
      for (int j = 0; j < input->dims()[1]; j++) {
        tmp_ptr[j] = mask_matrix_ptr[i * input->dims()[1] + j];
      }
    }
  }

  if (is_reverse) {
    mask_min_length = mask_min_length - time_step + 1;
  }

  bool has_allocate_mem_c = false;
  bool has_use_last_h_holder = false;
  const int& reverse_flag = is_reverse ? -1 : 1;

  // define the init_h holder for the swap
  Tensor init_h_temp;
  init_h_temp.Resize(init_h[layer_idx].dims());
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

  if ("LSTM" == mode) {
    last_c_holder = &(*last_c_ptr)[layer_idx];
    init_c_temp_holder = &init_c[layer_idx];
  } else if ("GRU" == mode) {
    // for reset output value
    last_c_temp.Resize(init_h[layer_idx].dims());
    last_c_temp.mutable_data<float16_t>();
    last_c_holder = &last_c_temp;
  }

  Tensor weight_hh_tmp;  // for gru
  std::vector<Tensor> weight_hh_tmp_ubind;
  std::vector<Tensor*> weight_hh_tmp_ubind_t;
  std::vector<int> stride_w;
  if ("GRU" == mode) {
    weight_hh_tmp.Resize(vec[1 + offset * 4].dims());
    weight_hh_tmp.mutable_data<float16_t>();
    weight_hh_tmp.CopyDataFrom(vec[1 + offset * 4]);
    int size = weight_hh_tmp.numel() / 3;
    std::memset(weight_hh_tmp.mutable_data<float16_t>() + size * 2,
                0,
                size * sizeof(float16_t));
  }

  for (int i = 0; i < time_step; i++) {
    bool in_mask = (reverse_flag * i) >= mask_min_length;
    if (i > 0) {
      if (!has_allocate_mem_c) {
        if (("LSTM" == mode) || ("GRU" == mode)) {
          init_c_temp.Resize(init_h[layer_idx].dims());
          init_c_temp.mutable_data<float16_t>();
          init_c_holder = &init_c_temp;
        }
        has_allocate_mem_c = true;
      }
      SwapPoniter(&init_c_holder, &last_c_holder);
      init_c_temp_holder = init_c_holder;
    }

    if ("LSTM" == mode) {
      lstm_cell_fp16(ctx,
                     &input_tensors[i],
                     &vec[1 + offset * 4],
                     init_h_holder,
                     init_c_temp_holder,
                     last_h_holder,
                     last_c_holder,
                     nullptr,
                     &output_tensors[i],
                     &vec[3 + offset * 4]);
    } else if ("GRU" == mode) {
      gru_cell_fp16(ctx,
                    &input_tensors[i],
                    &vec[1 + offset * 4],
                    init_h_holder,
                    init_c_temp_holder,
                    last_h_holder,
                    last_c_holder,
                    nullptr,
                    &output_tensors[i],
                    &vec[3 + offset * 4],
                    &weight_hh_tmp);
    }

    /*
     to be verifying!
    */
    if (in_mask) {
      postprocess_fp16(ctx,
                       &output_tensors[i],
                       init_h_holder,
                       init_c_temp_holder,
                       last_h_holder,
                       last_c_holder,
                       mask_vec[i],
                       mode);
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
      sd[i * st + j] = output_tensors[i].data<float16_t>()[j];
    }
  }

  if (has_sequence_length) {
    if (last_h_holder != &(*last_h_ptr)[layer_idx]) {
      (*last_h_ptr)[layer_idx].CopyDataFrom(*last_h_holder);
    }
  } else {
    (*last_h_ptr)[layer_idx].CopyDataFrom(output_tensors[time_step - 1]);
  }
  if ((0 == (time_step % 2)) && ("LSTM" == mode)) {
    (*last_c_ptr)[layer_idx].CopyDataFrom(*last_c_holder);
  }
}
#endif

template <>
void RnnCompute<PRECISION(kFloat)>::PrepareForRun() {}

template <>
void RnnCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::RnnParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
  param.Out->mutable_data<float>();
  std::string mode = param.mode;
  auto input = param.Input;
  auto weight_list = param.WeightList;
  auto pre_state = param.PreState;
  auto state = param.State;
  auto output = param.Out;
  bool is_bidirec = param.is_bidirec;
  int num_layers = param.num_layers;
  const Tensor* sequence_length = param.SequenceLength;
  int gate_num = 0;

  if ("LSTM" == mode) {
    gate_num = 4;
  } else if ("GRU" == mode) {
    gate_num = 3;
  } else {
    LOG(FATAL) << "ARM RNN ERROR: unsupport mode except gru and lstm,"
                  " present mode is "
               << mode;
    return;
  }

  state[0]->mutable_data<float>();
  if ("LSTM" == mode) {
    state[1]->mutable_data<float>();
  }

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
  init_h_unbind.resize(pre_state[0]->dims()[0]);
  last_h_unbind.resize(state[0]->dims()[0]);
  if ("LSTM" == mode) {
    init_c_unbind.resize(pre_state[1]->dims()[0]);
    last_c_unbind.resize(state[1]->dims()[0]);
  }

  std::vector<int> stride1, stride2;
  // unbind
  for (int i = 0; i < pre_state[0]->dims()[0]; i++) {
    stride1.push_back(1);
    int dim1 = pre_state[0]->dims()[1];
    int dim2 = pre_state[0]->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    init_h_unbind[i].Resize(dims);
    last_h_unbind[i].Resize(dims);
    init_h_unbind_t.push_back(&init_h_unbind[i]);
    last_h_unbind_t.push_back(&last_h_unbind[i]);
    last_h_unbind[i].mutable_data<float>();
  }

  lite::host::math::split(
      pre_state[0]->data<float>(), init_h_unbind_t, 0, stride1);

  if ("LSTM" == mode) {
    for (int i = 0; i < pre_state[1]->dims()[0]; i++) {
      stride2.push_back(1);
      int dim1 = pre_state[1]->dims()[1];
      int dim2 = pre_state[1]->dims()[2];
      DDimLite dims(std::vector<int64_t>{dim1, dim2});
      init_c_unbind[i].Resize(dims);
      last_c_unbind[i].Resize(dims);
      init_c_unbind_t.push_back(&init_c_unbind[i]);
      last_c_unbind_t.push_back(&last_c_unbind[i]);
      last_c_unbind[i].mutable_data<float>();
    }
    lite::host::math::split(
        pre_state[1]->data<float>(), init_c_unbind_t, 0, stride2);
  }

  std::vector<Tensor> output_vec(2);
  int time_step = input->dims()[0];
  int batch_size = input->dims()[1];
  int hidden_size = output->dims()[2];
  if (is_bidirec) {
    for (int i = 0; i < 2; ++i) {
      output_vec[i].Resize({time_step, batch_size, hidden_size / 2});
      output_vec[i].mutable_data<float>();
    }
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
      RUN_RNN_LAYER(i, &output_vec[0], true, 0);
      RUN_RNN_LAYER(i, &output_vec[1], true, 1);
      std::vector<Tensor*> output_vec_t = {&output_vec[0], &output_vec[1]};
      lite::arm::math::concat_func<float>(output_vec_t, 2, output_holder);
    } else {
      RUN_RNN_LAYER(i, output_holder, false, 0);
    }
  }

  lite::arm::math::concat_func<float>(last_h_unbind_t, 0, state[0]);
  if ("LSTM" == mode) {
    lite::arm::math::concat_func<float>(last_c_unbind_t, 0, state[1]);
  }

  // output_holder != output
  if (num_layers % 2 == 0) {
    output->CopyDataFrom(*output_holder);
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void RnnCompute<PRECISION(kFP16)>::PrepareForRun() {
  auto& param = this->Param<operators::RnnParam>();
  for (int i = 0; i < param.WeightList.size(); i++) {
    if (param.WeightList[i]->precision() != PRECISION(kFP16)) {
      Tensor tmp_tensor;
      tmp_tensor.CopyDataFrom(*(param.WeightList[i]));
      param.WeightList[i]->clear();
      param.WeightList[i]->set_precision(PRECISION(kFP16));
      float16_t* fp_data = param.WeightList[i]->mutable_data<float16_t>();
      const float* in_data = tmp_tensor.data<float>();
      lite::arm::math::fp16::fp32_to_fp16(
          in_data, fp_data, param.WeightList[i]->numel());
    }
  }
}

template <>
void RnnCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<operators::RnnParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
  param.Out->mutable_data<float16_t>();
  std::string mode = param.mode;
  auto input = param.Input;
  auto weight_list = param.WeightList;
  auto pre_state = param.PreState;
  auto state = param.State;
  auto output = param.Out;
  bool is_bidirec = param.is_bidirec;
  int num_layers = param.num_layers;
  const Tensor* sequence_length = param.SequenceLength;
  int gate_num = 0;

  if ("LSTM" == mode) {
    gate_num = 4;
  } else if ("GRU" == mode) {
    gate_num = 3;
  } else {
    LOG(FATAL) << "ARM RNN ERROR: unsupport mode except gru and lstm,"
                  " present mode is "
               << mode;
    return;
  }

  state[0]->mutable_data<float16_t>();
  if ("LSTM" == mode) {
    state[1]->mutable_data<float16_t>();
  }

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
  init_h_unbind.resize(pre_state[0]->dims()[0]);
  last_h_unbind.resize(state[0]->dims()[0]);
  if ("LSTM" == mode) {
    init_c_unbind.resize(pre_state[1]->dims()[0]);
    last_c_unbind.resize(state[1]->dims()[0]);
  }

  std::vector<int> stride1, stride2;
  // unbind
  for (int i = 0; i < pre_state[0]->dims()[0]; i++) {
    stride1.push_back(1);
    int dim1 = pre_state[0]->dims()[1];
    int dim2 = pre_state[0]->dims()[2];
    DDimLite dims(std::vector<int64_t>{dim1, dim2});
    init_h_unbind[i].Resize(dims);
    last_h_unbind[i].Resize(dims);
    init_h_unbind_t.push_back(&init_h_unbind[i]);
    last_h_unbind_t.push_back(&last_h_unbind[i]);
    last_h_unbind[i].mutable_data<float16_t>();
  }

  lite::host::math::split(
      pre_state[0]->data<float16_t>(), init_h_unbind_t, 0, stride1);

  if ("LSTM" == mode) {
    for (int i = 0; i < pre_state[1]->dims()[0]; i++) {
      stride2.push_back(1);
      int dim1 = pre_state[1]->dims()[1];
      int dim2 = pre_state[1]->dims()[2];
      DDimLite dims(std::vector<int64_t>{dim1, dim2});
      init_c_unbind[i].Resize(dims);
      last_c_unbind[i].Resize(dims);
      init_c_unbind_t.push_back(&init_c_unbind[i]);
      last_c_unbind_t.push_back(&last_c_unbind[i]);
      last_c_unbind[i].mutable_data<float16_t>();
    }
    lite::host::math::split(
        pre_state[1]->data<float16_t>(), init_c_unbind_t, 0, stride2);
  }

  std::vector<Tensor> output_vec(2);
  int time_step = input->dims()[0];
  int batch_size = input->dims()[1];
  int hidden_size = output->dims()[2];
  if (is_bidirec) {
    for (int i = 0; i < 2; ++i) {
      output_vec[i].Resize({time_step, batch_size, hidden_size / 2});
      output_vec[i].mutable_data<float16_t>();
    }
  }

  for (int i = 0; i < num_layers; i++) {
    if (i > 0) {
      if (!has_allocate_mem) {
        temp.Resize(output->dims());
        temp.mutable_data<float16_t>();
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
      RUN_RNN_LAYER_FP16(i, &output_vec[0], true, 0);
      RUN_RNN_LAYER_FP16(i, &output_vec[1], true, 1);
      std::vector<Tensor*> output_vec_t = {&output_vec[0], &output_vec[1]};
      lite::arm::math::concat_func<float16_t>(output_vec_t, 2, output_holder);
    } else {
      RUN_RNN_LAYER_FP16(i, output_holder, false, 0);
    }
  }

  lite::arm::math::concat_func<float16_t>(last_h_unbind_t, 0, state[0]);
  if ("LSTM" == mode) {
    lite::arm::math::concat_func<float16_t>(last_c_unbind_t, 0, state[1]);
  }

  // output_holder != output
  if (num_layers % 2 == 0) {
    output->CopyDataFrom(*output_holder);
  }
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
using rnn_f16_compute =
    paddle::lite::kernels::arm::RnnCompute<PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(rnn, kARM, kFP16, kNCHW, rnn_f16_compute, fp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("WeightList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("PreState",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("SequenceLength",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("DropoutState",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Reserve",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("State",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

using rnn_f32_compute =
    paddle::lite::kernels::arm::RnnCompute<PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(rnn, kARM, kFloat, kNCHW, rnn_f32_compute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("WeightList", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("PreState", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("SequenceLength",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("DropoutState", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Reserve", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("State", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
