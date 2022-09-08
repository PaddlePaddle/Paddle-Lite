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
  bool w_trans = param.transpose_w;
  enable_int8_ = param.enable_int8;
  per_channel_ = param.per_channel;
  quant_int16_ = param.enable_int16;
  CHECK(!(enable_int8_ && quant_int16_))
      << "param enable_int8 and enable_int16 can't be both true";
  // max
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  input_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));

  if (enable_int8_) {  // for paddle slim int8 quant
    output_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    xpu_quant_weight_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int8_t, int8_t>(
            reinterpret_cast<const int8_t*>(w_ptr),
            weight_dims,
            w_trans,
            per_channel_ ? param.weight_max.size() : max_ptr_size);
    CHECK(xpu_quant_weight_.max_ptr_ != nullptr)
        << "slim int8 quant xpu_quant_weight_max_ptr should't be null";
    std::vector<float> cpu_input_max(max_ptr_size, param.quant_input_max);
    lite::TargetWrapperXPU::MemcpySync(input_max_guard_->addr_,
                                       cpu_input_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    if (per_channel_) {
      lite::TargetWrapperXPU::MemcpySync(
          xpu_quant_weight_.max_ptr_,
          param.weight_max.data(),
          sizeof(float) * param.weight_max.size(),
          IoDirection::HtoD);
    } else {
      VLOG(3) << "set weight max :" << max_ptr_size
              << ", param.weight_max[0]:" << param.weight_max[0];
      std::vector<float> cpu_w_max(max_ptr_size, param.weight_max[0]);
      lite::TargetWrapperXPU::MemcpySync(xpu_quant_weight_.max_ptr_,
                                         cpu_w_max.data(),
                                         sizeof(float) * max_ptr_size,
                                         IoDirection::HtoD);
    }
    return;
  }

  if (quant_int16_) {
    xpu_quant_weight_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int16_t, int16_t>(
            reinterpret_cast<const int16_t*>(w_ptr),
            weight_dims,
            w_trans,
            max_ptr_size);
    std::vector<float> cpu_w_max(max_ptr_size, param.weight_max[0]);
    CHECK(xpu_quant_weight_.max_ptr_ != nullptr)
        << "slim int16 quant xpu_quant_weight_max_ptr should't be null";
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
  }

  xpu_quant_weight_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, TW>(
          w_ptr, weight_dims, w_trans, max_ptr_size);
  if (std::is_same<TW, float>::value) {
    VLOG(6) << "If fc compute precision is int31,must check weight max should "
               "be null ";
    CHECK(xpu_quant_weight_.max_ptr_ == nullptr)
        << "int31 weight max should be null";
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
  if (param.in_num_col_dims == -1) {
    param.in_num_col_dims += input_dims.size();
  }
  auto in_mat_dims = input_dims.Flatten2D(param.in_num_col_dims);
  int m = in_mat_dims[0];
  int k = in_mat_dims[1];
  int n = param.w->dims()[1];
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  param.output_max->Resize({max_ptr_size});

  bool x_trans = param.transpose_x;
  bool w_trans = param.transpose_w;
  int ldx = (x_trans ? m : k);
  int ldw = (w_trans ? k : n);
  int ldy = n;

  float* output_max =
      enable_int8_
          ? reinterpret_cast<float*>(output_max_guard_->addr_)
          : param.output_max->template mutable_data<float>(TARGET(kXPU));
  const auto* bias =
      param.has_bias ? param.bias->template data<float>() : nullptr;
  const float* input_max =
      (enable_int8_ || quant_int16_)
          ? reinterpret_cast<float*>(input_max_guard_->addr_)
          : (param.input_max ? param.input_max->template data<float>()
                             : nullptr);
  xdnn::Activation_t act((xdnn::Activation_t::act_enum)param.act_type);
  if (param.act_type == 5) {
    act.leaky_alpha = param.act_param;
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (param.act_type == 15) {
    act.hard_sigmoid_slope = param.act_param;
  }
  // TODO(weihaoji): remove fc_int31 and fc_int16 after xpu fc wrapper
  // refactor
  int r = 0;
  if (per_channel_ && !(std::is_same<TGEMM, float>::value)) {
    r = xdnn::fc_fusion_pc<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),                                       // ctx
        param.input->template data<DX>(),                          // x
        reinterpret_cast<const TW*>(xpu_quant_weight_.data_ptr_),  // w
        param.output->template mutable_data<DY>(TARGET(kXPU)),     // y
        m,                                                         // m
        n,                                                         // n
        k,                                                         // k
        x_trans,                                                   // x_trans
        w_trans,                                                   // w_trans
        input_max,                                                 // x_maxptr
        nullptr,                                                   // w_maxptr
        output_max,                                                // y_maxptr
        ldx,                                                       // ldx
        ldw,                                                       // ldw
        ldy,                                                       // ldy
        1.0f,                                                      // alpha
        0.0f,                                                      // beta
        bias,                                                      // bias
        reinterpret_cast<const float*>(
            xpu_quant_weight_.max_ptr_),  // per channel weight_max
        act);
  } else {
    r = xdnn::fc_fusion<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),                                         // ctx
        param.input->template data<DX>(),                            // x
        reinterpret_cast<const TW*>(xpu_quant_weight_.data_ptr_),    // w
        param.output->template mutable_data<DY>(TARGET(kXPU)),       // y
        m,                                                           // m
        n,                                                           // n
        k,                                                           // k
        x_trans,                                                     // x_trans
        w_trans,                                                     // w_trans
        input_max,                                                   // x_maxptr
        reinterpret_cast<const float*>(xpu_quant_weight_.max_ptr_),  // w_maxptr
        output_max,                                                  // y_maxptr
        ldx,                                                         // ldx
        ldw,                                                         // ldw
        ldy,                                                         // ldy
        1.0f,                                                        // alpha
        0.0f,                                                        // beta
        bias,                                                        // bias
        act);
  }
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

using XPUFC_Int8_FP32_FP32 =
    xpu::XPUFcCompute<int8_t, int8_t, float, float, PRECISION(kInt8)>;

using XPUFC_FP32_LOCAL_QUANT =
    xpu::XPUFcCompute<float, float, float, float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kFloat, kNCHW, XPUFC_FP32, XPU_Real_kFloat)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__fc, kXPU, kFloat, kNCHW, XPUFC_FP16_FP32_FP32, def)
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

REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kInt8, kNCHW, XPUFC_Int8_FP32_FP32, XPU_Int8_FP32_FP32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUFC_Int8_Int8_Int8 =
    xpu::XPUFcCompute<int8_t, int8_t, int8_t, int8_t, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kInt8, kNCHW, XPUFC_Int8_Int8_Int8, XPU_Int8_Int8_Int8)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUFC_Int8_Int8_FP32 =
    xpu::XPUFcCompute<int8_t, int8_t, int8_t, float, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(
    __xpu__fc, kXPU, kInt8, kNCHW, XPUFC_Int8_Int8_FP32, XPU_Int8_Int8_FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__fc,
                     kXPU,
                     kFloat,
                     kNCHW,
                     XPUFC_FP32_LOCAL_QUANT,
                     XPU_FP32_LOCAL_QUANT)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
