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

#include "lite/kernels/loongarch/interpolate_compute.h"
#include <string>
#include <vector>
#include "lite/backends/loongarch/math/interpolate.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

void BilinearInterpCompute::Run() {
  auto& param = Param<operators::InterpolateParam>();
  // required input
  lite::Tensor* X = param.X;
  // optionla inputs
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  // output
  lite::Tensor* Out = param.Out;
  // optional attributes
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  int align_mode = param.align_mode;
  // required attributes
  bool align_corners = param.align_corners;
  std::string interp_method = "Bilinear";
  lite::loongarch::math::interpolate(X,
                               OutSize,
                               SizeTensor,
                               Scale,
                               Out,
                               scale,
                               param.scale_v,
                               out_h,
                               out_w,
                               align_mode,
                               align_corners,
                               interp_method);
}

void NearestInterpCompute::Run() {
  auto& param = Param<operators::InterpolateParam>();
  // required input
  lite::Tensor* X = param.X;
  // optionla inputs
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  // output
  lite::Tensor* Out = param.Out;
  // optional attributes
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  int align_mode = param.align_mode;
  // required attributes
  bool align_corners = param.align_corners;
  std::string interp_method = "Nearest";
  lite::loongarch::math::interpolate(X,
                               OutSize,
                               SizeTensor,
                               Scale,
                               Out,
                               scale,
                               param.scale_v,
                               out_h,
                               out_w,
                               align_mode,
                               align_corners,
                               interp_method);
}

void NearestInterpComputeV2::Run() {
  auto& param = Param<operators::InterpolateParam>();
  // required input
  lite::Tensor* X = param.X;
  // optionla inputs
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  // output
  lite::Tensor* Out = param.Out;
  // optional attributes
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  int align_mode = param.align_mode;
  // required attributes
  bool align_corners = param.align_corners;
  std::string interp_method = "Nearest";
  lite::loongarch::math::interpolate_v2(X,
                                  OutSize,
                                  SizeTensor,
                                  Scale,
                                  Out,
                                  scale,
                                  param.scale_v,
                                  out_h,
                                  out_w,
                                  align_mode,
                                  align_corners,
                                  interp_method);
}

}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(bilinear_interp,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::BilinearInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::NearestInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp_v2,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::BilinearInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp_v2,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::NearestInterpComputeV2,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();
