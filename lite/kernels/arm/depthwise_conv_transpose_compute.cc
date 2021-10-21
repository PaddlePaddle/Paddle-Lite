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

#include "lite/kernels/arm/depthwise_conv_transpose_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::
    DepthwiseConv2DTransposeCompute<PRECISION(kFloat), PRECISION(kFloat)>
        DepConvTransFp32;
typedef paddle::lite::kernels::arm::
    DepthwiseConv2DTransposeCompute<PRECISION(kInt8), PRECISION(kFloat)>
        DepConvTranInt8_Fp32;
typedef paddle::lite::kernels::arm::
    DepthwiseConv2DTransposeCompute<PRECISION(kInt8), PRECISION(kInt8)>
        DepConvTranInt8_Int8;

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::
    DepthwiseConv2DTransposeCompute<PRECISION(kFP16), PRECISION(kFP16)>
        DepConvTranFp16;

REGISTER_LITE_KERNEL(
    depthwise_conv2d_transpose, kARM, kFP16, kNCHW, DepConvTranFp16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();

#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(
    depthwise_conv2d_transpose, kARM, kFloat, kNCHW, DepConvTransFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d_transpose,
                     kARM,
                     kInt8,
                     kNCHW,
                     DepConvTranInt8_Fp32,
                     fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d_transpose,
                     kARM,
                     kInt8,
                     kNCHW,
                     DepConvTranInt8_Int8,
                     int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();
