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
  local_quant_ = std::is_same<TGEMM, float>::value;
  CHECK_LE(static_cast<int>(enable_int8_) + static_cast<int>(quant_int16_) +
               static_cast<int>(local_quant_),
           1)
      << "enable_int8/enable_int16/local_quant at most 1 be set";
  if (enable_int8_ && param.quant_output_max == 0) {
    CHECK((std::is_same<DY, float>::value))
        << "when out scale = 0; fc output precision must float.";
  }

  // max
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  if (!local_quant_) {
    input_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
  }

  if (enable_int8_) {  // for paddle slim int8 quant
    output_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    xpu_quant_weight_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int8_t, int8_t>(
            reinterpret_cast<const int8_t*>(w_ptr),
            weight_dims,
            !w_trans,
            per_channel_ ? param.weight_max.size() : max_ptr_size);
    CHECK(xpu_quant_weight_.max_ptr_ != nullptr)
        << "slim int8 quant xpu_quant_weight_max_ptr should't be null";
    std::vector<float> cpu_output_max(max_ptr_size, param.quant_output_max);
    lite::TargetWrapperXPU::MemcpySync(output_max_guard_->addr_,
                                       cpu_output_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);

    if (per_channel_) {
      weight_one_value_guard_ =
          TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    }
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
      std::vector<float> cpu_weight_one_value(max_ptr_size, 1.0);
      lite::TargetWrapperXPU::MemcpySync(weight_one_value_guard_->addr_,
                                         cpu_weight_one_value.data(),
                                         sizeof(float) * max_ptr_size,
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
            !w_trans,
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
          w_ptr, weight_dims, !w_trans, max_ptr_size);
  if (std::is_same<TW, float>::value) {
    // if TW is fp16, max_ptr is not necessary, however the
    // ConvertCPUWeightToXPUQuantWeight generated it, so we just ignored it
    CHECK(xpu_quant_weight_.max_ptr_ == nullptr)
        << "float weight max should be null";
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
  if (w_trans) {
    n = param.w->dims()[0];
  }
  int ldx = (x_trans ? m : k);
  int ldw = k;
  int ldy = n;

  float* output_max =
      enable_int8_
          ? param.quant_output_max
                ? reinterpret_cast<float*>(output_max_guard_->addr_)
                : nullptr
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
  float* pc_weight_max_ptr = nullptr;
  float* weight_max_ptr = nullptr;
  if (per_channel_ && !(std::is_same<TGEMM, float>::value)) {
    pc_weight_max_ptr = reinterpret_cast<float*>(xpu_quant_weight_.max_ptr_);
    weight_max_ptr = reinterpret_cast<float*>(weight_one_value_guard_->addr_);
  } else {
    weight_max_ptr = reinterpret_cast<float*>(xpu_quant_weight_.max_ptr_);
  }
  if (local_quant_) {
    int64_t batch_size = input_dims[0];
    CHECK_GT(batch_size, 0) << "first dim shouldn't be zero";
    std::vector<int> m_lod(batch_size + 1);
    for (int i = 0; i < batch_size + 1; ++i) {
      m_lod[i] = i * m / batch_size;
    }
    r = xdnn::fc_fusion_norm<DX, TW, DY, int>(
        ctx.GetRawContext(),                                       // ctx
        param.input->template data<DX>(),                          // x
        reinterpret_cast<const TW*>(xpu_quant_weight_.data_ptr_),  // w
        param.output->template mutable_data<DY>(TARGET(kXPU)),     // y
        m,                                                         // m
        n,                                                         // n
        k,                                                         // k
        x_trans,                                                   // x_trans
        true,                                                      // w_trans
        input_max,                                                 // x_maxptr
        weight_max_ptr,                                            // w_maxptr
        output_max,                                                // y_maxptr
        ldx,                                                       // ldx
        ldw,                                                       // ldw
        ldy,                                                       // ldy
        param.alpha,                                               // alpha
        0.0f,                                                      // beta
        bias,                                                      // bias
        act,                                                       // act
        1.0f,
        batch_size,
        m_lod);  // per channel weight_max
  } else {
    r = xdnn::fc_fusion<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),                                       // ctx
        param.input->template data<DX>(),                          // x
        reinterpret_cast<const TW*>(xpu_quant_weight_.data_ptr_),  // w
        param.output->template mutable_data<DY>(TARGET(kXPU)),     // y
        m,                                                         // m
        n,                                                         // n
        k,                                                         // k
        x_trans,                                                   // x_trans
        true,                                                      // w_trans
        input_max,                                                 // x_maxptr
        weight_max_ptr,                                            // w_maxptr
        output_max,                                                // y_maxptr
        ldx,                                                       // ldx
        ldw,                                                       // ldw
        ldy,                                                       // ldy
        param.alpha,                                               // alpha
        0.0f,                                                      // beta
        bias,                                                      // bias
        act,                                                       // act
        pc_weight_max_ptr);  // per channel weight_max
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
    xpu::XPUFcCompute<float, float16, float, float, PRECISION(kFloat)>;

using XPUFC_Int8_Int8_FP32 =
    xpu::XPUFcCompute<int8_t, int8_t, int8_t, float, PRECISION(kInt8)>;

using XPUFC_Int8_Int8_Int8 =
    xpu::XPUFcCompute<int8_t, int8_t, int8_t, int8_t, PRECISION(kInt8)>;

using XPUFC_Int8_Int8_FP32_Int8 =
    xpu::XPUFcCompute<int8_t, int8_t, float, int8_t, PRECISION(kInt8)>;

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

REGISTER_LITE_KERNEL(__xpu__fc,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUFC_Int8_Int8_FP32_Int8,
                     XPU_Int8_Int8_FP32_Int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
