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

#include "lite/kernels/metal/image_op/depthwise_conv2d_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(depthwise_conv2d,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::DepthwiseConv2dImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Filter",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Output",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::DepthwiseConv2dImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Filter",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Output",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
