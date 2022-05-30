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

template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
void XPUFcCompute<TGEMM, TW, DX, DY, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  auto w_ptr = param.w->template data<float>();
  auto weight_dims = param.w->dims();
  bool quant_int8 = false;
  if (param.quant_w_max > 0.f) {
    quant_int8 = true;
  }
  // max
  int max_ptr_size = xdnn::get_max_ptr_size(ctx.GetRawContext());
  param.output_max->Resize({max_ptr_size});
  input_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
  if (quant_int8) {  // for paddle slim int8 quant
    xpu_quant_weight_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int8_t, int8_t>(
            reinterpret_cast<const int8_t*>(w_ptr), weight_dims, true);
    std::vector<float> cpu_w_max(max_ptr_size, param.quant_w_max);
    CHECK(xpu_quant_weight_.max_ptr_ != nullptr)
        << "slim int8 quant xpu_quant_weight_max_ptr should't be null";
    lite::TargetWrapperXPU::MemcpySync(xpu_quant_weight_.max_ptr_,
                                       cpu_w_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    std::vector<float> cpu_input_max(max_ptr_size, param.quant_input_max);
    lite::TargetWrapperXPU::MemcpySync(input_max_guard_->addr_,
                                       cpu_input_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    return;
  } else {
    xpu_quant_weight_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, TW>(
            w_ptr, weight_dims, true);
    if (std::is_same<TW, float>::value) {
      VLOG(6)
          << "If fc compute precision is int31,must check weight max should "
             "be null ";
      CHECK(xpu_quant_weight_.max_ptr_ == nullptr)
          << "int31 weight max should be null";
    }
  }
}
template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
void XPUFcCompute<TGEMM, TW, DX, DY, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto input_dims = param.input->dims();
  auto in_mat_dims = input_dims.Flatten2D(param.in_num_col_dims);
  int m = in_mat_dims[0];
  int k = in_mat_dims[1];
  int n = param.w->dims()[1];
  bool quant_int8 = param.quant_w_max > 0.f;

  float* output_max =
      quant_int8 ? nullptr
                 : param.output_max->template mutable_data<float>(TARGET(kXPU));
  const auto* bias =
      param.has_bias ? param.bias->template data<float>() : nullptr;
  const float* input_max =
      quant_int8 ? reinterpret_cast<float*>(input_max_guard_->addr_)
                 : (param.input_max ? param.input_max->template data<float>()
                                    : nullptr);
  xdnn::Activation_t act((xdnn::Activation_t::act_enum)param.act_type);
  if (param.act_type == 5) {
    act.leaky_alpha = param.act_param;
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (param.act_type == 15) {
    act.hard_sigmoid_slope = param.act_param;
  }
  // TODO(weihaoji): remove fc_int31 and fc_int16 after xpu fc wrapper refactor

  int r = xdnn::fc_fusion<DX, TW, DY, TGEMM>(
      ctx.GetRawContext(),                                         // ctx
      param.input->template data<DX>(),                            // x
      reinterpret_cast<const TW*>(xpu_quant_weight_.data_ptr_),    // w
      param.output->template mutable_data<DY>(TARGET(kXPU)),       // y
      m,                                                           // m
      n,                                                           // n
      k,                                                           // k
      false,                                                       // x_trans
      true,                                                        // w_trans
      input_max,                                                   // x_maxptr
      reinterpret_cast<const float*>(xpu_quant_weight_.max_ptr_),  // w_maxptr
      output_max,                                                  // y_maxptr
      k,                                                           // ldx
      k,                                                           // ldw
      n,                                                           // ldy
      1.0f,                                                        // alpha
      0.0f,                                                        // beta
      bias,                                                        // bias
      act);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUFC_FP32 =
    xpu::XPUFcCompute<int, float, float, float, PRECISION(kFloat)>;

using XPUFC_FP16_FP32_FP32 =
    xpu::XPUFcCompute<int16_t, int16_t, float, float, PRECISION(kFloat)>;

using XPUFC_FP16_FP16_FP16 =
    xpu::XPUFcCompute<int16_t, int16_t, float16, float16, PRECISION(kFP16)>;

using XPUFC_FP16_FP32_FP16 =
    xpu::XPUFcCompute<int16_t, int16_t, float, float16, PRECISION(kFP16)>;

using XPUFC_FP16_FP16_FP32 =
    xpu::XPUFcCompute<int16_t, int16_t, float16, float, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kFloat, kNCHW, XPUFC_FP32, XPU_Real_kFloat)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kFloat, kNCHW, XPUFC_FP16_FP32_FP32, XPUFC_FP16_FP32_FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kFP16, kNCHW, XPUFC_FP16_FP16_FP16, XPUFC_FP16_FP16_FP16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kFP16, kNCHW, XPUFC_FP16_FP32_FP16, XPUFC_FP16_FP32_FP16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kFP16, kNCHW, XPUFC_FP16_FP16_FP32, XPUFC_FP16_FP16_FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
