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

#include "lite/kernels/xpu/__xpu__fc_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUFcCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto w_ptr = param.w->data<float>();
  auto w_len = param.w->numel();
  auto weight_dims = param.w->dims();
  // max
  w_max = paddle::lite::xpu::math::FindMaxAbs(w_ptr, w_len);
  std::vector<float> w_max_v(4, w_max);
  weight_max_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(weight_max_guard_->addr_),
                      w_max_v.data(),
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  // transpose
  std::vector<float> transpose_w(w_len, 0);
  paddle::lite::xpu::math::Transpose(
      w_ptr, transpose_w.data(), weight_dims[0], weight_dims[1]);
  // quant
  if (param.precision == "int31") {
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(float));
    XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(quant_weight_guard_->addr_),
                        transpose_w.data(),
                        w_len * sizeof(float),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  } else if (param.precision == "int16") {
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(int16_t));
    std::vector<int16_t> quant_weight_cpu(w_len, 0);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        transpose_w.data(), quant_weight_cpu.data(), w_max, w_len);
    XPU_CALL(xpu_memcpy(reinterpret_cast<int16_t*>(quant_weight_guard_->addr_),
                        quant_weight_cpu.data(),
                        w_len * sizeof(int16_t),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  } else if (param.precision == "int8") {
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(int8_t));
    std::vector<int8_t> quant_weight_cpu(w_len, 0);
    paddle::lite::xpu::math::ConvertFP32ToInt8(
        transpose_w.data(), quant_weight_cpu.data(), w_max, w_len);
    XPU_CALL(xpu_memcpy(reinterpret_cast<int8_t*>(quant_weight_guard_->addr_),
                        quant_weight_cpu.data(),
                        w_len * sizeof(int8_t),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  }
  input_max_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  output_max_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
}

void XPUFcCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto input_dims = param.input->dims();
  param.in_mat_dims = input_dims.Flatten2D(param.in_num_col_dims);
  int m = param.in_mat_dims[0];
  int k = param.in_mat_dims[1];
  int n = param.w->dims()[1];
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;
  xdnn::Activation_t act_type = (param.activation_type == "relu")
                                    ? xdnn::Activation_t::RELU
                                    : xdnn::Activation_t::LINEAR;

  int r = -1;
  if (param.precision == "int31") {
    r = xdnn::fc_int31(ctx.GetRawContext(),        /* context */
                       false,                      /* TransA */
                       param.transpose_w,          /* TransB */
                       m,                          /* m */
                       n,                          /* n */
                       k,                          /* k */
                       1.0f,                       /* alpha */
                       param.input->data<float>(), /* A */
                       nullptr,                    /* max_a ptr */
                       param.w->data<float>(),     /* B */
                       param.w_max,                /* max_b */
                       0.0f,                       /* beta */
                       param.output->mutable_data<float>(TARGET(kXPU)), /* C */
                       nullptr, /* max_c ptr */
                       bias,    /* bias */
                       act_type /* act_type */);
  } else {
    r = xdnn::findmax<float>(ctx.GetRawContext(),
                             param.input->data<float>(),
                             m * k,
                             reinterpret_cast<float*>(input_max_guard_->addr_));
    CHECK_EQ(r, 0);
    r = xdnn::fc_int16(
        ctx.GetRawContext(),        /* context */
        false,                      /* TransA */
        true,                       /* TransB */
        m,                          /* m */
        n,                          /* n */
        k,                          /* k */
        1.0f,                       /* alpha */
        param.input->data<float>(), /* A */
        reinterpret_cast<const float*>(input_max_guard_->addr_),
        reinterpret_cast<const int16_t*>(quant_weight_guard_->addr_), /* B */
        reinterpret_cast<const float*>(weight_max_guard_->addr_),
        0.0f,                                            /* beta */
        param.output->mutable_data<float>(TARGET(kXPU)), /* C */
        reinterpret_cast<float*>(output_max_guard_->addr_),
        bias, /* bias */
        act_type /* act_type */);
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__fc,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUFcCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
