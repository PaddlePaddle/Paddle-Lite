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

#include "lite/kernels/xpu/gru_compute.h"
#include <string>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline xdnn::Activation_t GetActType(const std::string& act_type) {
  if (act_type == "sigmoid") {
    return xdnn::Activation_t::SIGMOID;
  } else if (act_type == "tanh") {
    return xdnn::Activation_t::TANH;
  } else if (act_type == "relu") {
    return xdnn::Activation_t::RELU;
  } else {
    LOG(FATAL) << "unsupported activation type:" << act_type;
    return xdnn::Activation_t::LINEAR;
  }
}

void GRUCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto weight = param.weight;
  auto weight_ptr = weight->data<float>();
  auto weight_dims = weight->dims();
  int frame_size = weight_dims[0];
  int weight_len = weight->numel();
  // ptr and len
  float* weight_s1_ptr = const_cast<float*>(weight_ptr);
  int weight_s1_len = frame_size * 2 * frame_size;
  float* weight_s2_ptr = weight_s1_ptr + weight_s1_len;
  int weight_s2_len = frame_size * frame_size;
  CHECK_EQ(weight_len, weight_s1_len + weight_s2_len);
  // max
  weight_s1_abs_max_ =
      paddle::lite::xpu::math::FindMaxAbs(weight_s1_ptr, weight_s1_len);
  weight_s2_abs_max_ =
      paddle::lite::xpu::math::FindMaxAbs(weight_s2_ptr, weight_s2_len);
  std::vector<float> weight_max_vector(8);
  for (int i = 0; i < 4; i++) {
    weight_max_vector[i] = weight_s1_abs_max_;
    weight_max_vector[i + 4] = weight_s2_abs_max_;
  }
  weight_max_guard_ = TargetWrapperXPU::MallocScratchPad(8 * sizeof(float));
  XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(weight_max_guard_->addr_),
                      weight_max_vector.data(),
                      8 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  // quant
  quant_weight_guard_ =
      TargetWrapperXPU::MallocScratchPad(weight_len * sizeof(int16_t));
  std::vector<int16_t> quant_weight_cpu(weight_len);
  int16_t* quant_weight_s1_cpu_ptr = quant_weight_cpu.data();
  int16_t* quant_weight_s2_cpu_ptr = quant_weight_s1_cpu_ptr + weight_s1_len;
  paddle::lite::xpu::math::ConvertFP32ToInt16(weight_s1_ptr,
                                              quant_weight_s1_cpu_ptr,
                                              weight_s1_abs_max_,
                                              weight_s1_len);
  paddle::lite::xpu::math::ConvertFP32ToInt16(weight_s2_ptr,
                                              quant_weight_s2_cpu_ptr,
                                              weight_s2_abs_max_,
                                              weight_s2_len);
  XPU_CALL(xpu_memcpy(reinterpret_cast<int16_t*>(quant_weight_guard_->addr_),
                      quant_weight_cpu.data(),
                      weight_len * sizeof(int16_t),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void GRUCompute::Run() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = *param_.get_mutable<operators::GRUParam>();

  bool origin_mode = param.origin_mode;
  bool is_reverse = param.is_reverse;
  auto gate_activation = GetActType(param.gate_activation);
  auto activation = GetActType(param.activation);

  auto* input = param.input;
  const float* input_ptr = input->data<float>();

  auto* h0 = param.h0;
  const float* hidden_prev_ptr = (h0 == nullptr) ? nullptr : h0->data<float>();

  const int16_t* weight_ptr =
      reinterpret_cast<const int16_t*>(quant_weight_guard_->addr_);
  const float* weight_maxptr =
      reinterpret_cast<const float*>(weight_max_guard_->addr_);

  auto* bias = param.bias;
  const float* bias_ptr = (bias == nullptr) ? nullptr : bias->data<float>();

  auto* hidden = param.hidden;
  float* hidden_ptr = hidden->mutable_data<float>(TARGET(kXPU));
  const auto& hidden_dims = hidden->dims();
  int frame_size = hidden_dims[1];

  auto& input_lod = input->lod()[0];
  int ret = xdnn::gru_core<float, int16_t, float, int16_t>(ctx.GetRawContext(),
                                                           input_ptr,
                                                           hidden_prev_ptr,
                                                           weight_ptr,
                                                           hidden_ptr,
                                                           input_lod,
                                                           frame_size,
                                                           nullptr,
                                                           nullptr,
                                                           weight_maxptr,
                                                           nullptr,
                                                           bias_ptr,
                                                           activation,
                                                           gate_activation,
                                                           origin_mode,
                                                           is_reverse);
  CHECK_EQ(ret, 0) << "call xdnn::gru_core failed!";
  // batch_gate, batch_reset_hidden_prev lod not set
  hidden->set_lod(input->lod());
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    gru, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::GRUCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
