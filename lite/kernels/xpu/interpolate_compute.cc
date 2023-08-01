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

#include "lite/kernels/xpu/interpolate_compute.h"
#include <iostream>
#include <memory>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void BilinearInterpCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  lite::Tensor* Out = param.Out;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int align_mode = param.align_mode;
  bool align_corners = param.align_corners;

  int trans_mode = -1;
  if (align_corners == true) {
    trans_mode = 0;
  } else if ((align_corners == false) && (align_mode == 0)) {
    trans_mode = 1;
  } else {
    trans_mode = 2;
  }

  float* quant_input_max =
      param.enable_int8
          ? reinterpret_cast<float*>(quant_input_max_value_guard_->addr_)
          : nullptr;
  float* quant_output_max =
      param.enable_int8
          ? reinterpret_cast<float*>(quant_output_max_value_guard_->addr_)
          : nullptr;

  int r = xdnn::interpolate2d<InType>(ctx.GetRawContext(),
                                      X->data<InType>(),
                                      Out->mutable_data<InType>(TARGET(kXPU)),
                                      n,
                                      c,
                                      in_h,
                                      in_w,
                                      out_h,
                                      out_w,
                                      false,
                                      trans_mode,
                                      true,
                                      quant_input_max,
                                      quant_output_max);
  CHECK_EQ(r, 0);
}

template <typename InType, PrecisionType PType>
void BilinearInterpCompute<InType, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  if (param.enable_int8) {
    quant_input_max_value_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    std::vector<float> cpu_quant_input_max_value(max_ptr_size,
                                                 param.input_scale);
    lite::TargetWrapperXPU::MemcpySync(quant_input_max_value_guard_->addr_,
                                       cpu_quant_input_max_value.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);

    quant_output_max_value_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    std::vector<float> cpu_quant_output_max_value(max_ptr_size,
                                                  param.output_scale);
    lite::TargetWrapperXPU::MemcpySync(quant_output_max_value_guard_->addr_,
                                       cpu_quant_output_max_value.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
  }
}

template <typename InType, PrecisionType PType>
void NearestInterpCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  lite::Tensor* Out = param.Out;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  bool align_corners = param.align_corners;
  int trans_mode = (align_corners == true) ? 0 : 2;

  float* quant_input_max =
      param.enable_int8
          ? reinterpret_cast<float*>(quant_input_max_value_guard_->addr_)
          : nullptr;
  float* quant_output_max =
      param.enable_int8
          ? reinterpret_cast<float*>(quant_output_max_value_guard_->addr_)
          : nullptr;

  int r = xdnn::interpolate2d<InType>(ctx.GetRawContext(),
                                      X->data<InType>(),
                                      Out->mutable_data<InType>(TARGET(kXPU)),
                                      n,
                                      c,
                                      in_h,
                                      in_w,
                                      out_h,
                                      out_w,
                                      true,
                                      trans_mode,
                                      true,
                                      quant_input_max,
                                      quant_output_max);

  CHECK_EQ(r, 0);
}

template <typename InType, PrecisionType PType>
void NearestInterpCompute<InType, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  if (param.enable_int8) {
    quant_input_max_value_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    std::vector<float> cpu_quant_input_max_value(max_ptr_size,
                                                 param.input_scale);
    lite::TargetWrapperXPU::MemcpySync(quant_input_max_value_guard_->addr_,
                                       cpu_quant_input_max_value.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);

    quant_output_max_value_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    std::vector<float> cpu_quant_output_max_value(max_ptr_size,
                                                  param.output_scale);
    lite::TargetWrapperXPU::MemcpySync(quant_output_max_value_guard_->addr_,
                                       cpu_quant_output_max_value.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
  }
}

template <typename InType, PrecisionType PType>
void BicubicInterpCompute<InType, PType>::PrepareForRun() {}

template <typename InType, PrecisionType PType>
void BicubicInterpCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  lite::Tensor* Out = param.Out;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int align_mode = param.align_mode;
  bool align_corners = param.align_corners;

  int trans_mode = -1;
  if (align_corners == true) {
    trans_mode = 0;
  } else if ((align_corners == false) && (align_mode == 0)) {
    trans_mode = 1;
  } else {
    trans_mode = 2;
  }

  int r =
      xdnn::bicubic_resize2d<InType>(ctx.GetRawContext(),
                                     X->data<InType>(),
                                     Out->mutable_data<InType>(TARGET(kXPU)),
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     out_h,
                                     out_w,
                                     trans_mode,
                                     true);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using BiliInterp_FP32 = xpu::BilinearInterpCompute<float, PRECISION(kFloat)>;
using BiliInterp_FP16 = xpu::BilinearInterpCompute<float16, PRECISION(kFP16)>;
using BiliInterp_INT8 = xpu::BilinearInterpCompute<int8_t, PRECISION(kInt8)>;

using NearInterp_FP32 = xpu::NearestInterpCompute<float, PRECISION(kFloat)>;
using NearInterp_FP16 = xpu::NearestInterpCompute<float16, PRECISION(kFP16)>;
using NearInterp_INT8 = xpu::NearestInterpCompute<int8_t, PRECISION(kInt8)>;

using BicubicInterp_FP32 = xpu::BicubicInterpCompute<float, PRECISION(kFloat)>;
using BicubicInterp_FP16 = xpu::BicubicInterpCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(bilinear_interp, kXPU, kFloat, kNCHW, BiliInterp_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp,
                     kXPU,
                     kFP16,
                     kNCHW,
                     BiliInterp_FP16,
                     DISABLE_XPU1_binterp_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    bilinear_interp, kXPU, kInt8, kNCHW, BiliInterp_INT8, binterp_INT8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    bilinear_interp_v2, kXPU, kFloat, kNCHW, BiliInterp_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp_v2,
                     kXPU,
                     kFP16,
                     kNCHW,
                     BiliInterp_FP16,
                     DISABLE_XPU1_binterp_v2_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    bilinear_interp_v2, kXPU, kInt8, kNCHW, BiliInterp_INT8, binterp_v2_INT8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp, kXPU, kFloat, kNCHW, NearInterp_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
                     kXPU,
                     kFP16,
                     kNCHW,
                     NearInterp_FP16,
                     DISABLE_XPU1_ninterp_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp, kXPU, kInt8, kNCHW, NearInterp_INT8, ninterp_INT8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp_v2, kXPU, kFloat, kNCHW, NearInterp_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp_v2,
                     kXPU,
                     kFP16,
                     kNCHW,
                     NearInterp_FP16,
                     DISABLE_XPU1_niterp_v2_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp_v2, kXPU, kInt8, kNCHW, NearInterp_INT8, niterp_v2_INT8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    bicubic_interp_v2, kXPU, kFloat, kNCHW, BicubicInterp_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(bicubic_interp_v2,
                     kXPU,
                     kFP16,
                     kNCHW,
                     BicubicInterp_FP16,
                     DISABLE_XPU1_bicubicinterp_v2_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
