// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/conv3d_compute.h"
#include <vector>
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
void Conv3DCompute<TGEMM, TW, DX, DY, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  auto filter_ptr = param.filter->template data<float>();
  auto filter_dims = param.filter->dims();
  xpu_quant_filter_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, TW>(
          filter_ptr, filter_dims, false, ctx.GetRawContext()->max_ptr_size());
}

template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
void Conv3DCompute<TGEMM, TW, DX, DY, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& x_dims = param.x->dims();
  auto& w_dims = param.filter->dims();
  int groups = param.groups;
  auto& strides = param.strides;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  int r = xdnn::conv3d<DX, TW, DY, TGEMM>(
      ctx.GetRawContext(), /* context */
      param.x->template data<DX>(),
      reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_), /* weight */
      param.output->template mutable_data<DY>(TARGET(kXPU)),
      x_dims[0], /* input_n */
      x_dims[1], /* input_c */
      x_dims[2], /* input_d */
      x_dims[3], /* input_h */
      x_dims[4], /* input_w */
      w_dims[0], /* num_filter */
      std::vector<int>{static_cast<int>(w_dims[2]),
                       static_cast<int>(w_dims[3]),
                       static_cast<int>(w_dims[4])}, /* kernel size*/
      strides,
      paddings,
      dilations,
      groups,
      nullptr,
      reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
      nullptr,
      true /*is_ncdhw*/);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUConv3dFP32 =
    xpu::Conv3DCompute<int, float, float, float, PRECISION(kFloat)>;

using XPUConv3d_FP16_FP32_FP32 =
    xpu::Conv3DCompute<int16_t, int16_t, float, float, PRECISION(kFloat)>;

using XPUConv3dFp16 =
    xpu::Conv3DCompute<int16_t, int16_t, float16, float16, PRECISION(kFP16)>;

using XPUConv3d_FP16_FP16_FP32 =
    xpu::Conv3DCompute<int16_t, int16_t, float16, float, PRECISION(kFP16)>;

using XPUConv3d_FP16_FP32_FP16 =
    xpu::Conv3DCompute<int16_t, int16_t, float, float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    conv3d, kXPU, kFloat, kNCHW, XPUConv3dFP32, XPU_Real_kFloat)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(conv3d, kXPU, kFloat, kNCHW, XPUConv3d_FP16_FP32_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv3d, kXPU, kFP16, kNCHW, XPUConv3dFp16, XPU_FP16_FP16_FP16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv3d, kXPU, kFP16, kNCHW, XPUConv3d_FP16_FP16_FP32, XPU_FP16_FP16_FP32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv3d, kXPU, kFP16, kNCHW, XPUConv3d_FP16_FP32_FP16, XPU_FP16_FP32_FP16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
