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

#include "lite/kernels/xpu/__xpu__conv2d_compute.h"
#include "lite/backends/xpu/math.h"
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
void XPUConv2dCompute<TGEMM, TW, DX, DY, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  param.output_max->Resize({max_ptr_size});
  auto filter_ptr = param.filter->template data<float>();
  auto filter_dims = param.filter->dims();
  bool quant_int8 = param.enable_int8;
  bool quant_int16 = param.enable_int16;
  CHECK(!(quant_int8 && quant_int16))
      << "param enable_int8 and enable_int16 can't be both true";
  if (quant_int8 || quant_int16) {
    input_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    if (quant_int8) {
      output_max_guard_ =
          TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
      branch_max_guard_ =
          TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    }
  }

  // for paddle slim int8 quant
  if (quant_int8) {
    xpu_quant_filter_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int8_t, int8_t>(
            reinterpret_cast<const int8_t*>(filter_ptr),
            filter_dims,
            false,
            max_ptr_size);
    std::vector<float> cpu_w_max(max_ptr_size, param.quant_w_max);
    CHECK(xpu_quant_filter_.max_ptr_ != nullptr)
        << "slim int8 quant xpu_quant_filter_max_ptr should't be null";
    lite::TargetWrapperXPU::MemcpySync(xpu_quant_filter_.max_ptr_,
                                       cpu_w_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    std::vector<float> cpu_input_max(max_ptr_size, param.quant_input_max);
    lite::TargetWrapperXPU::MemcpySync(input_max_guard_->addr_,
                                       cpu_input_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    std::vector<float> cpu_output_max(max_ptr_size, param.quant_output_max);
    lite::TargetWrapperXPU::MemcpySync(output_max_guard_->addr_,
                                       cpu_output_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    if (param.quant_branch_max != 0) {
      std::vector<float> cpu_branch_max(max_ptr_size, param.quant_branch_max);
      lite::TargetWrapperXPU::MemcpySync(branch_max_guard_->addr_,
                                         cpu_branch_max.data(),
                                         sizeof(float) * max_ptr_size,
                                         IoDirection::HtoD);
    }
    return;
  }

  // for paddle slim int16 quant
  if (quant_int16) {
    xpu_quant_filter_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<int16_t, int16_t>(
            reinterpret_cast<const int16_t*>(filter_ptr),
            filter_dims,
            false,
            max_ptr_size);
    std::vector<float> cpu_w_max(max_ptr_size, param.quant_w_max);
    CHECK(xpu_quant_filter_.max_ptr_ != nullptr)
        << "slim int16 quant xpu_quant_filter_max_ptr should't be null";
    lite::TargetWrapperXPU::MemcpySync(xpu_quant_filter_.max_ptr_,
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

  // data precision cast FP16 <-> FP32
  xpu_quant_filter_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, TW>(
          filter_ptr, filter_dims, false, max_ptr_size);
}

template <
    typename DY,
    typename std::enable_if<!xdnn::is_int8<DY>(), DY>::type* ptr = nullptr>
static int broadcastadd(xdnn::Context* ctx,
                        const DY* x,
                        const DY* y,
                        DY* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) {
  int r = xdnn::broadcast_add<DY>(ctx, x, y, z, xshape, yshape);
  return r;
}

template <typename DY,
          typename std::enable_if<xdnn::is_int8<DY>(), DY>::type* ptr = nullptr>
static int broadcastadd(xdnn::Context* ctx,
                        const DY* x,
                        const DY* y,
                        DY* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) {
  return xdnn::NOT_IMPLEMENT;
}

template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
void XPUConv2dCompute<TGEMM, TW, DX, DY, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& input_dims = param.input->dims();
  auto& filter_dims = param.filter_dims;
  int batch = static_cast<int>(input_dims[0]);
  int img_c = static_cast<int>(input_dims[1]);
  int img_h = static_cast<int>(input_dims[2]);
  int img_w = static_cast<int>(input_dims[3]);
  int filter_num = static_cast<int>(filter_dims[0]);
  int win_h = static_cast<int>(filter_dims[2]);
  int win_w = static_cast<int>(filter_dims[3]);
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int groups = param.groups.front();
  int act_type = param.act_type.front();
  bool quant_int8 = param.enable_int8;
  bool quant_int16 = param.enable_int16;

  const float* input_max =
      (quant_int8 || quant_int16)
          ? reinterpret_cast<float*>(input_max_guard_->addr_)
          : (param.input_max ? param.input_max->template data<float>()
                             : nullptr);
  // output_max,branch_maxs only set  when use int8
  float* output_max =
      quant_int8 ? reinterpret_cast<float*>(output_max_guard_->addr_)
                 : param.output_max->template mutable_data<float>(TARGET(kXPU));
  float* branch_max = (quant_int8 && (param.quant_branch_max != 0))
                          ? reinterpret_cast<float*>(branch_max_guard_->addr_)
                          : nullptr;
  const auto* bias =
      param.has_bias ? param.bias->template data<float>() : nullptr;
  const DY* branch =
      param.has_branch ? param.branch->template data<DY>() : nullptr;

  xdnn::Activation_t act((xdnn::Activation_t::act_enum)act_type);
  if (act_type == 5) {
    act.leaky_alpha = param.act_param.front();
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (act_type == 15) {
    act.hard_sigmoid_slope = param.act_param.front();
  }
  if (branch != nullptr && param.output->dims() != param.branch->dims()) {
    CHECK_EQ(act_type, 0);
    if (branch_broadcast_guard_.get() == nullptr) {
      branch_broadcast_guard_ = TargetWrapperXPU::MallocScratchPad(
          param.output->numel() * sizeof(DY));
    } else {
      branch_broadcast_guard_->Reserve(param.output->numel() * sizeof(DY));
    }

    int r = xdnn::conv2d_fusion<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),
        param.input->template data<DX>(),
        reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_),
        reinterpret_cast<DY*>(branch_broadcast_guard_->addr_),
        batch,
        img_c,
        img_h,
        img_w,
        filter_num,
        std::vector<int>{win_h, win_w},
        param.strides,
        paddings,
        dilations,
        groups,
        input_max,
        reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
        output_max,
        true,
        bias,
        nullptr,
        act);

    CHECK_EQ(r, 0);

    auto conv_out_shape = param.output->dims().Vectorize();
    auto branch_shape = param.branch->dims().Vectorize();
    std::vector<int> xshape =
        std::vector<int>(conv_out_shape.begin(), conv_out_shape.end());
    std::vector<int> yshape =
        std::vector<int>(branch_shape.begin(), branch_shape.end());
    if (branch_shape > conv_out_shape) {
      param.output->Resize(lite::DDim(branch_shape));
    }
    DY* output = param.output->template mutable_data<DY>(TARGET(kXPU));
    r = broadcastadd<DY>(ctx.GetRawContext(),
                         reinterpret_cast<DY*>(branch_broadcast_guard_->addr_),
                         branch,
                         output,
                         xshape,
                         yshape);
    CHECK_EQ(r, 0);
  } else {
    DY* output = param.output->template mutable_data<DY>(TARGET(kXPU));
    int r = xdnn::conv2d_fusion<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),
        param.input->template data<DX>(),
        reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_),
        output,
        batch,
        img_c,
        img_h,
        img_w,
        filter_num,
        std::vector<int>{win_h, win_w},
        param.strides,
        paddings,
        dilations,
        groups,
        input_max,
        reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
        output_max,
        true,
        bias,
        branch,
        act,
        branch_max);
    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using XPUConv2d_FP16_FP32_FP32 =
    xpu::XPUConv2dCompute<int16_t, int16_t, float, float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    __xpu__conv2d, kXPU, kFloat, kNCHW, XPUConv2d_FP16_FP32_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dFP32 =
    xpu::XPUConv2dCompute<int, float, float, float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    __xpu__conv2d, kXPU, kFloat, kNCHW, XPUConv2dFP32, XPU_Real_kFloat)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dFp16 =
    xpu::XPUConv2dCompute<int16_t, int16_t, float16, float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    __xpu__conv2d, kXPU, kFP16, kNCHW, XPUConv2dFp16, XPU_FP16_FP16__FP16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2d_FP16_FP16_FP32 =
    xpu::XPUConv2dCompute<int16_t, int16_t, float16, float, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kFP16,
                     kNCHW,
                     XPUConv2d_FP16_FP16_FP32,
                     XPU_FP16_FP16__FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2d_FP16_FP32_FP16 =
    xpu::XPUConv2dCompute<int16_t, int16_t, float, float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kFP16,
                     kNCHW,
                     XPUConv2d_FP16_FP32_FP16,
                     XPU_FP16_FP32__FP16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dInt8_FP32_FP32 =
    xpu::XPUConv2dCompute<int8_t, int8_t, float, float, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUConv2dInt8_FP32_FP32,
                     XPU_Int8_FP32_FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dInt8_Int8_Int8 =
    xpu::XPUConv2dCompute<int8_t, int8_t, int8_t, int8_t, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUConv2dInt8_Int8_Int8,
                     XPU_Int8_Int8_Int8)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dInt8_FP32_Int8 =
    xpu::XPUConv2dCompute<int8_t, int8_t, float, int8_t, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUConv2dInt8_FP32_Int8,
                     XPU_Int8_FP32_Int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dInt8_Int8_FP32 =
    xpu::XPUConv2dCompute<int8_t, int8_t, int8_t, float, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUConv2dInt8_Int8_FP32,
                     XPU_Int8_Int8_FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUConv2dInt8_FP16_FP16 =
    xpu::XPUConv2dCompute<int8_t, int8_t, float16, float16, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUConv2dInt8_FP16_FP16,
                     XPU_Int8_FP16_FP16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
