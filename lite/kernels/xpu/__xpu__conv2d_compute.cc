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

template <typename T>
bool QuantFilter(const float* filter_on_host,
                 T* quant_res,
                 float max,
                 int64_t len) {
  return false;
}

template <>
bool QuantFilter<int16_t>(const float* filter_on_host,
                          int16_t* quant_res,
                          float max,
                          int64_t len) {
  paddle::lite::xpu::math::ConvertFP32ToInt16(
      filter_on_host, quant_res, max, len);
  return true;
}

template <>
bool QuantFilter<int8_t>(const float* filter_on_host,
                         int8_t* quant_res,
                         float max,
                         int64_t len) {
  paddle::lite::xpu::math::ConvertFP32ToInt8(
      filter_on_host, quant_res, max, len);
  return true;
}

template <typename T, PrecisionType PType>
void XPUConv2dCompute<T, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  param.output_max->Resize({max_ptr_size});
  auto filter_ptr = param.filter->template data<float>();
  auto filter_dims = param.filter->dims();

  xpu_quant_filter_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, T>(
          filter_ptr, filter_dims, false);
}

template <typename T, PrecisionType PType>
void XPUConv2dCompute<T, PType>::Run() {
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
  float* output_max =
      param.output_max->template mutable_data<float>(TARGET(kXPU));
  const auto* bias =
      param.has_bias ? param.bias->template data<float>() : nullptr;
  const float* branch =
      param.has_branch ? param.branch->template data<float>() : nullptr;
  const float* input_max =
      param.input_max ? param.input_max->template data<float>() : nullptr;
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
          param.output->numel() * sizeof(float));
    } else {
      branch_broadcast_guard_->Reserve(param.output->numel() * sizeof(float));
    }
    int r = xdnn::conv2d_fusion<float, T, float, T>(
        ctx.GetRawContext(),
        param.input->template data<float>(),
        reinterpret_cast<const T*>(xpu_quant_filter_.data_ptr_),
        reinterpret_cast<float*>(branch_broadcast_guard_->addr_),
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
    float* output = param.output->template mutable_data<float>(TARGET(kXPU));
    r = xdnn::broadcast_add<float>(
        ctx.GetRawContext(),
        reinterpret_cast<float*>(branch_broadcast_guard_->addr_),
        branch,
        output,
        xshape,
        yshape);
    CHECK_EQ(r, 0);
  } else {
    float* output = param.output->template mutable_data<float>(TARGET(kXPU));
    int r = xdnn::conv2d_fusion<float, T, float, T>(
        ctx.GetRawContext(),
        param.input->template data<float>(),
        reinterpret_cast<const T*>(xpu_quant_filter_.data_ptr_),
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
        act);
    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using XPUConv2dFp32 = xpu::XPUConv2dCompute<int16_t, PRECISION(kFloat)>;

using XPUConv2dInt8 = xpu::XPUConv2dCompute<int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(__xpu__conv2d, kXPU, kFloat, kNCHW, XPUConv2dFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__conv2d, kXPU, kInt8, kNCHW, XPUConv2dInt8, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
