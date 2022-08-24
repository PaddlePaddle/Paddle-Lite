// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__squeeze_excitation_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUSqueezeExcitationCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto weight_ptr = param.filter->data<float>();
  auto weight_len = param.filter->numel();
  auto weight1_len = weight_len / 2;
  auto weight2_len = weight_len / 2;
  auto weight1_dims = paddle::lite::DDimLite();
  auto weight2_dims = paddle::lite::DDimLite();
  weight1_dims.ConstructFrom({weight1_len});
  weight2_dims.ConstructFrom({weight2_len});
  auto max_ptr_len = ctx.GetRawContext()->max_ptr_size();
  quant_weight1_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
          weight_ptr, weight1_dims, false, max_ptr_len);
  quant_weight2_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
          weight_ptr + weight1_len, weight2_dims, false, max_ptr_len);
}

void XPUSqueezeExcitationCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto input_dims = param.input->dims();
  int batch = input_dims[0];
  int channel = input_dims[1];
  int h = input_dims[2];
  int w = input_dims[3];
  const auto* branch =
      param.has_branch ? param.branch->template data<float>() : nullptr;
  auto filter_dims = param.filter_dims;
  const float* bias1_ptr =
      param.has_bias ? param.bias->template data<float>() : nullptr;
  const float* bias2_ptr =
      (bias1_ptr != nullptr)
          ? (bias1_ptr + param.filter_dims[1] / param.filter_dims[0])
          : nullptr;

  std::vector<xdnn::Activation_t> act;
  for (size_t i = 0; i < 3; i++) {
    xdnn::Activation_t cur_act =
        (xdnn::Activation_t::act_enum)param.act_type[i];
    if (param.act_type[i] == 5) {
      cur_act.leaky_alpha = param.act_param[i];
      CHECK(cur_act.leaky_alpha >= 0.0001 && cur_act.leaky_alpha <= 10);
    } else if (param.act_type[i] == 15) {
      cur_act.hard_sigmoid_slope = param.act_param[i];
    }
    act.push_back(cur_act);
  }
  int r = xdnn::squeeze_excitation_block<float, int16_t, int16_t>(
      ctx.GetRawContext(),
      param.input->data<float>(),
      reinterpret_cast<const int16_t*>(quant_weight1_.data_ptr_),
      reinterpret_cast<const int16_t*>(quant_weight2_.data_ptr_),
      param.output->mutable_data<float>(TARGET(kXPU)),
      batch,
      channel,
      h,
      w,
      filter_dims[0],
      reinterpret_cast<float*>(quant_weight1_.max_ptr_),
      reinterpret_cast<float*>(quant_weight2_.max_ptr_),
      bias1_ptr,
      bias2_ptr,
      branch,
      act[0],
      act[1],
      act[2]);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__squeeze_excitation_block,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUSqueezeExcitationCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
