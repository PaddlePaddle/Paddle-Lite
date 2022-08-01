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

#include "lite/kernels/arm/interpolate_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define INIT_PARAM(method_name)                       \
  auto& param = Param<operators::InterpolateParam>(); \
  lite::Tensor* X = param.X;                          \
  lite::Tensor* OutSize = param.OutSize;              \
  auto SizeTensor = param.SizeTensor;                 \
  auto Scale = param.Scale;                           \
  lite::Tensor* Out = param.Out;                      \
  float scale = param.scale;                          \
  int out_w = param.out_w;                            \
  int out_h = param.out_h;                            \
  bool align_corners = param.align_corners;           \
  int align_mode = param.align_mode;                  \
  auto scale_v = param.scale_v;                       \
  std::string interp_method = method_name;

#define INTERP_PARAM                                                      \
  X, OutSize, SizeTensor, Scale, Out, out_h, out_w, scale, align_corners, \
      align_mode, interp_method, scale_v

template <>
void BilinearInterpCompute<PRECISION(kFloat)>::Run() {
  INIT_PARAM("Bilinear")
  lite::arm::math::interpolate(INTERP_PARAM);
}

template <>
void NearestInterpCompute<PRECISION(kFloat)>::Run() {
  INIT_PARAM("Nearest")
  lite::arm::math::interpolate(INTERP_PARAM);
}

template <>
void NearestInterpComputeV2<PRECISION(kFloat)>::Run() {
  INIT_PARAM("Nearest")
  lite::arm::math::nearest_interp_v2<float>(INTERP_PARAM);
}

#ifdef ENABLE_ARM_FP16
template <>
void BilinearInterpCompute<PRECISION(kFP16)>::Run() {
  INIT_PARAM("Bilinear")
  lite::arm::math::fp16::interpolate(INTERP_PARAM);
}

template <>
void NearestInterpCompute<PRECISION(kFP16)>::Run() {
  INIT_PARAM("Nearest")
  lite::arm::math::fp16::interpolate(INTERP_PARAM);
}

template <>
void NearestInterpComputeV2<PRECISION(kFP16)>::Run() {
  INIT_PARAM("Nearest")
  lite::arm::math::nearest_interp_v2<float16_t>(INTERP_PARAM);
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::BilinearInterpCompute<PRECISION(kFP16)>
    bilinear_interp_fp16;
typedef paddle::lite::kernels::arm::NearestInterpCompute<PRECISION(kFP16)>
    nearest_interp_fp16;
typedef paddle::lite::kernels::arm::NearestInterpComputeV2<PRECISION(kFP16)>
    nearest_interp_v2_fp16;

REGISTER_LITE_KERNEL(
    bilinear_interp, kARM, kFP16, kNCHW, bilinear_interp_fp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp, kARM, kFP16, kNCHW, nearest_interp_fp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    bilinear_interp_v2, kARM, kFP16, kNCHW, bilinear_interp_fp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp_v2, kARM, kFP16, kNCHW, nearest_interp_v2_fp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

typedef paddle::lite::kernels::arm::BilinearInterpCompute<PRECISION(kFloat)>
    bilinear_interp_fp32;
typedef paddle::lite::kernels::arm::NearestInterpCompute<PRECISION(kFloat)>
    nearest_interp_fp32;

typedef paddle::lite::kernels::arm::NearestInterpComputeV2<PRECISION(kFloat)>
    nearest_interp_v2_fp32;

REGISTER_LITE_KERNEL(
    bilinear_interp, kARM, kFloat, kNCHW, bilinear_interp_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp, kARM, kFloat, kNCHW, nearest_interp_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    bilinear_interp_v2, kARM, kFloat, kNCHW, bilinear_interp_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp_v2, kARM, kFloat, kNCHW, nearest_interp_v2_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
