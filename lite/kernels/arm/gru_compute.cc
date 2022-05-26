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

#include "lite/kernels/arm/gru_compute.h"
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
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/gru_utils_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

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

void GRUComputeRun(const operators::GRUParam& param,
                   ARMContext* ctx,
                   bool enable_quant) {
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

  // memory
  batch_reset_hidden_prev->Resize(hidden_dims);
  batch_hidden->Resize(hidden_dims);
  float* batch_gate_data = batch_gate->mutable_data<float>();
  batch_reset_hidden_prev->mutable_data<float>();
  batch_hidden->mutable_data<float>();
  hidden->mutable_data<float>();
  memset(batch_gate_data, 0, batch_gate->numel() * sizeof(float));

  std::vector<float> weight_scale{};
  int bit_length{};
  if (enable_quant) {
    CHECK(param.enable_int8);
    CHECK_EQ(weight->dims().size(), 2);
    CHECK_EQ(param.weight_scale.size(), weight->dims()[1]);
    weight_scale = param.weight_scale;
    bit_length = param.bit_length;
  }

  lite::arm::math::LoDTensor2BatchFunctor<float> to_batch;
  to_batch(*input, batch_gate, true, param.is_reverse);

  if (bias) {
    auto bias_data = bias->data<float>();
    lite::arm::math::gru_add_with_bias(batch_gate_data,
                                       bias_data,
                                       batch_gate_data,
                                       batch_size,
                                       frame_size * 3);
  }

  lite::arm::math::GRUMetaValue<float> gru_value;
  if (enable_quant) {
    if (weight->precision() != PRECISION(kInt8)) {
      LOG(FATAL) << "Precision Error: The precision of quantized gru's "
                 << "weights should be int8_t, but it is "
                 << static_cast<int>(weight->precision());
    }
    const int8_t* weight_data = weight->data<int8_t>();
    gru_value.gate_weight_int8 = const_cast<int8_t*>(weight_data);
    gru_value.state_weight_int8 =
        const_cast<int8_t*>(weight_data + 2 * frame_size * frame_size);
  } else {
    const float* weight_data = weight->data<float>();
    gru_value.gate_weight = const_cast<float*>(weight_data);
    gru_value.state_weight =
        const_cast<float*>(weight_data + 2 * frame_size * frame_size);
  }

  Tensor ordered_h0;
  std::vector<uint64_t> order(batch_gate->lod()[2]);
  if (h0) {
    // Since the batch computing for GRU reorders the input sequences
    // according to their length. The initialized cell state also needs
    // to reorder.
    lite::arm::math::ReorderInitState<float>(*h0, order, &ordered_h0, true);
    gru_value.prev_out_value = ordered_h0.mutable_data<float>();
  } else {
    gru_value.prev_out_value = nullptr;
  }

  auto batch_starts = batch_gate->lod()[0];
  size_t seq_len = batch_starts.size() - 1;
  auto active_node = get_gru_act_type(param.activation);
  auto active_gate = get_gru_act_type(param.gate_activation);

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

    if (enable_quant) {
      lite::arm::math::GRUUnitFunctor<float>::quant_compute(gru_value,
                                                            frame_size,
                                                            cur_batch_size,
                                                            active_node,
                                                            active_gate,
                                                            param.origin_mode,
                                                            weight_scale,
                                                            bit_length,
                                                            ctx);
    } else {
      lite::arm::math::GRUUnitFunctor<float>::compute(gru_value,
                                                      frame_size,
                                                      cur_batch_size,
                                                      active_node,
                                                      active_gate,
                                                      param.origin_mode,
                                                      ctx);
    }
    gru_value.prev_out_value = gru_value.output_value;
  }
  lite::arm::math::Batch2LoDTensorFunctor<float> to_seq;
  *(batch_hidden->mutable_lod()) = batch_gate->lod();
  to_seq(*batch_hidden, hidden);
}

template <>
void GRUCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::GRUParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
  GRUComputeRun(param, &ctx, false);
}

template <>
void GRUCompute<PRECISION(kInt8)>::Run() {
  auto& param = this->Param<operators::GRUParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
  GRUComputeRun(param, &ctx, true);
}

#ifdef ENABLE_ARM_FP16
template <>
void GRUCompute<PRECISION(kFP16)>::Run() {
  auto& param = this->Param<operators::GRUParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
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

  // memory
  batch_reset_hidden_prev->Resize(hidden_dims);
  batch_hidden->Resize(hidden_dims);
  float16_t* batch_gate_data = batch_gate->mutable_data<float16_t>();
  batch_reset_hidden_prev->mutable_data<float16_t>();
  batch_hidden->mutable_data<float16_t>();
  hidden->mutable_data<float16_t>();
  memset(batch_gate_data, 0, batch_gate->numel() * sizeof(float16_t));

  lite::arm::math::LoDTensor2BatchFunctor<float16_t> to_batch;
  to_batch(*input, batch_gate, true, param.is_reverse);

  if (bias) {
    auto bias_data = bias->data<float16_t>();
    lite::arm::math::fp16::gru_add_with_bias(batch_gate_data,
                                             bias_data,
                                             batch_gate_data,
                                             batch_size,
                                             frame_size * 3);
  }
  lite::arm::math::fp16::GRUMetaValue<float16_t> gru_value;
  const float16_t* weight_data = weight->data<float16_t>();
  gru_value.gate_weight = const_cast<float16_t*>(weight_data);
  gru_value.state_weight =
      const_cast<float16_t*>(weight_data + 2 * frame_size * frame_size);
  Tensor ordered_h0;
  std::vector<uint64_t> order(batch_gate->lod()[2]);
  if (h0) {
    // Since the batch computing for GRU reorders the input sequences
    // according to their length. The initialized cell state also needs
    // to reorder.
    lite::arm::math::ReorderInitState<float16_t>(*h0, order, &ordered_h0, true);
    gru_value.prev_out_value = ordered_h0.mutable_data<float16_t>();
  } else {
    gru_value.prev_out_value = nullptr;
  }

  auto batch_starts = batch_gate->lod()[0];
  size_t seq_len = batch_starts.size() - 1;
  auto active_node = get_gru_act_type(param.activation);
  auto active_gate = get_gru_act_type(param.gate_activation);

  for (size_t n = 0; n < seq_len; n++) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    int cur_batch_size = bend - bstart;

    gru_value.output_value = batch_hidden->mutable_data<float16_t>() +
                             bstart * batch_hidden->dims()[1];
    gru_value.gate_value =
        batch_gate->mutable_data<float16_t>() + bstart * batch_gate->dims()[1];
    gru_value.reset_output_value =
        batch_reset_hidden_prev->mutable_data<float16_t>() +
        bstart * batch_reset_hidden_prev->dims()[1];

    lite::arm::math::fp16::GRUUnitFunctor<float16_t>::compute(gru_value,
                                                              frame_size,
                                                              cur_batch_size,
                                                              active_node,
                                                              active_gate,
                                                              param.origin_mode,
                                                              &ctx);

    gru_value.prev_out_value = gru_value.output_value;
  }
  lite::arm::math::Batch2LoDTensorFunctor<float16_t> to_seq;
  *(batch_hidden->mutable_lod()) = batch_gate->lod();
  to_seq(*batch_hidden, hidden);
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(gru,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::GRUCompute<PRECISION(kFP16)>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Weight",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("BatchGate",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("BatchResetHiddenPrev",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("BatchHidden",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Hidden",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(gru,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GRUCompute<PRECISION(kFloat)>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(gru,
                     kARM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::arm::GRUCompute<PRECISION(kInt8)>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
