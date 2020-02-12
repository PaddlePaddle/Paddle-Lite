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
#include <unistd.h>

#include <iostream>
#include <string>
#include <vector>

#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/gru_utils.h"
#include "lite/backends/arm/math/sequence2batch.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/kernels/fpga/gru_compute.h"

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/backends/fpga/KD/pes/gru_util.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

inline lite_api::ActivationType get_gru_act_type(const std::string& type) {
  if (type == "sigmoid") {
    return lite_api::ActivationType::kSigmoid;
  } else if (type == "tanh") {
    return lite_api::ActivationType::kTanh;
  } else if (type == "relu") {
    return lite_api::ActivationType::kRelu;
  } else if (type == "identity") {
    return lite_api::ActivationType::kIndentity;
  } else {
    LOG(FATAL) << "unsupported activation type: " << type;
  }
}

void GRUCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  param.hidden->mutable_data<float>();

  auto input = param.input;
  auto h0 = param.h0;
  auto weight = param.weight;
  auto bias = param.bias;

  zynqmp::GRUParam& gru_param = pe_.param();
  gru_param.input = input->ZynqTensor();
  if (h0 != nullptr) {
    gru_param.h0 = h0->ZynqTensor();
  }
  gru_param.weight = weight->ZynqTensor();
  gru_param.bias = bias->ZynqTensor();

  gru_param.batch_gate = param.batch_gate->ZynqTensor();
  gru_param.batch_reset_hidden_prev =
      param.batch_reset_hidden_prev->ZynqTensor();
  gru_param.batch_hidden = param.batch_hidden->ZynqTensor();
  gru_param.hidden = param.hidden->ZynqTensor();

  gru_param.gate_activation = param.gate_activation;
  gru_param.activation = param.activation;

  pe_.init();
  pe_.apply();
}

void GRUCompute::Run() {
  auto& param = this->Param<param_t>();
  param.hidden->mutable_data<float>();

  // inputs
  auto input = param.input;
  auto h0 = param.h0;
  auto weight = param.weight;
  auto bias = param.bias;
  // outputs
  auto batch_gate = param.batch_gate;
  auto batch_reset_hidden_prev = param.batch_reset_hidden_prev;
  auto batch_hidden = param.batch_hidden;
  auto hidden = param.hidden;

  auto hidden_dims = hidden->dims();
  int frame_size = hidden_dims[1];
  auto batch_size = input->dims()[0];

  const float* weight_data = weight->data<float>();
  float* batch_gate_data = batch_gate->mutable_data<float>();

  lite::arm::math::LoDTensor2BatchFunctor<float> to_batch;
  to_batch(*input, batch_gate, true, param.is_reverse);  // 1.

  if (bias) {
    auto bias_data = bias->data<float>();  // 2.
    lite::arm::math::gru_add_with_bias(batch_gate_data,
                                       bias_data,
                                       batch_gate_data,
                                       batch_size,
                                       frame_size * 3);
  }

  zynqmp::GRUTensors gru_tensors;
  lite::arm::math::GRUMetaValue<float> gru_value;
  gru_value.gate_weight = const_cast<float*>(weight_data);
  gru_value.state_weight =
      const_cast<float*>(weight_data + 2 * frame_size * frame_size);

  Tensor ordered_h0;
  std::vector<uint64_t> order(batch_gate->lod()[2]);

  if (h0) {
    // Since the batch computing for GRU reorders the input sequences
    // according to their length. The initialized cell state also needs
    // to reorder.
    // lite::arm::math::ReorderInitState<float>(*h0, order, &ordered_h0, true);
    // //3.
    gru_value.prev_out_value = ordered_h0.mutable_data<float>();
    gru_tensors.pre_output = ordered_h0.ZynqTensor();

  } else {
    gru_value.prev_out_value = nullptr;
    gru_tensors.pre_output = nullptr;
  }
  auto batch_starts = batch_gate->lod()[0];
  size_t seq_len = batch_starts.size() - 1;
  auto active_node = get_gru_act_type(param.activation);
  auto active_gate = get_gru_act_type(param.gate_activation);

  save_float(gru_value.gate_weight, "_gate_weight.txt", weight->numel());
  batch_gate->ZynqTensor()->saveToFile("batch_gate.txt");

  zynqmp::Tensor float_input;
  zynqmp::Tensor hidden_out;

  for (size_t n = 0; n < seq_len; n++) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    int cur_batch_size = bend - bstart;

    gru_value.output_value =
        batch_hidden->mutable_data<float>() + bstart * batch_hidden->dims()[1];
    gru_value.gate_value =
        batch_gate->mutable_data<float>() + bstart * batch_gate->dims()[1];
    gru_value.reset_output_value =
        batch_reset_hidden_prev->mutable_data<float>() +
        bstart * batch_reset_hidden_prev->dims()[1];

    zynqmp::Shape float_input_shape(zynqmp::NC,
                                    {cur_batch_size, batch_gate->dims()[1]});
    float* float_data =
        float_input.mutableData<float>(zynqmp::FP32, float_input_shape);
    memcpy(float_data,
           gru_value.gate_value,
           batch_gate->dims()[1] * sizeof(float));
    float_input.flush();

    float* hidden_data =
        hidden_out.mutableData<float>(zynqmp::FP32, float_input_shape);

    gru_tensors.gate = &float_input;
    gru_tensors.output = &hidden_out;

    pe_.GRUCOmpute(gru_tensors,
                   frame_size,
                   cur_batch_size,
                   active_node,
                   active_gate,
                   param.origin_mode);

    // TODO(chonwhite): copy data back to original tensor;

    gru_tensors.pre_output = gru_tensors.output;
  }
  lite::arm::math::Batch2LoDTensorFunctor<float> to_seq;  // 5.
  *(batch_hidden->mutable_lod()) = batch_gate->lod();
  batch_hidden->mutable_data<float>();
  to_seq(*batch_hidden, hidden);
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    gru, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::GRUCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
