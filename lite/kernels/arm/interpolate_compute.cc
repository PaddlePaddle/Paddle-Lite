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

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void BilinearInterpCompute::Run() {
  auto& param = Param<operators::InterpolateParam>();
  lite::Tensor* X = param.X;
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  lite::Tensor* Out = param.Out;
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  bool align_corners = param.align_corners;
  std::string interp_method = "Bilinear";
  lite::arm::math::interpolate(X,
                               OutSize,
                               SizeTensor,
                               Scale,
                               Out,
                               out_h,
                               out_w,
                               scale,
                               align_corners,
                               interp_method);
}

void NearestInterpCompute::Run() {
  auto& param = Param<operators::InterpolateParam>();
  lite::Tensor* X = param.X;
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  lite::Tensor* Out = param.Out;
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  bool align_corners = param.align_corners;
  std::string interp_method = "Nearest";
  lite::arm::math::interpolate(X,
                               OutSize,
                               SizeTensor,
                               Scale,
                               Out,
                               out_h,
                               out_w,
                               scale,
                               align_corners,
                               interp_method);
}

} /* namespace arm */
} /* namespace kernels */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_KERNEL(bilinear_interp,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::BilinearInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NearestInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
