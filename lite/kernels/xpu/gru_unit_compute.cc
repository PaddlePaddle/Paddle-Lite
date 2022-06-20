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

#include "lite/kernels/xpu/gru_unit_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GRUUnitCompute::PrepareForRun() {
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

void GRUUnitCompute::Run() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = *param_.get_mutable<operators::GRUUnitParam>();
  // inputs
  auto input = param.input;
  auto hidden_prev = param.hidden_prev;
  auto bias = param.bias;
  // outputs
  auto hidden = param.hidden;
  // args
  int gate_activation = param.gate_activation;
  CHECK_EQ(gate_activation, 1)
      << "Only support gate_activation=1(sigmoid) but received "
      << gate_activation << " in XPU gru_unit kernel";
  int activation = param.activation;
  CHECK_EQ(activation, 2) << "Only support activation=2(tanh) but received "
                          << activation << " in XPU gru_unit kernel";
  bool origin_mode = param.origin_mode;

  int batch_size = input->dims()[0];
  int frame_size = hidden_prev->dims()[1];

  const float* input_ptr = input->data<float>();
  const float* hidden_prev_ptr = hidden_prev->data<float>();
  const int16_t* weight_ptr =
      reinterpret_cast<const int16_t*>(quant_weight_guard_->addr_);
  const float* weight_maxptr =
      reinterpret_cast<const float*>(weight_max_guard_->addr_);
  const float* bias_ptr = (bias == nullptr) ? nullptr : bias->data<float>();

  float* hidden_ptr = hidden->mutable_data<float>(TARGET(kXPU));

  int ret = xdnn::gru_unit<float, int16_t, float, int16_t>(
      ctx.GetRawContext(),
      input_ptr,
      hidden_prev_ptr,
      weight_ptr,
      hidden_ptr,
      batch_size,
      frame_size,
      nullptr,
      nullptr,
      weight_maxptr,
      nullptr,
      bias_ptr,
      xdnn::Activation_t::TANH,
      xdnn::Activation_t::SIGMOID,
      origin_mode);
  CHECK_EQ(ret, 0) << "call xdnn::gru_unit failed!";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gru_unit,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GRUUnitCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("HiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Gate", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("ResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
