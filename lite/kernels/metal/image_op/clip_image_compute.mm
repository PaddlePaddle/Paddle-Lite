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

#include "lite/kernels/metal/image_op/clip_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void ClipImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.out->dims();
    auto input_dims = param.x->dims();

    lite::Tensor* min_tensor = param.min_tensor;
    lite::Tensor* max_tensor = param.max_tensor;
    min_ = param.min;
    max_ = param.max;

    if (min_tensor != nullptr) {
        min_ = min_tensor->data<float>()[0];
    }
    if (max_tensor != nullptr) {
        max_ = max_tensor->data<float>()[0];
    }
#ifdef LITE_WITH_METAL_FULL
#else
    output_buffer_ = param.out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
    input_buffer_ = param.x->data<MetalHalf, MetalImage>();
#endif

    setup_without_mps();
}

void ClipImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void ClipImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_->image() atIndex:(0)];
    [encoder setTexture:output_buffer_->image() atIndex:(1)];
    [encoder setBuffer:params_buffer_->buffer() offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void ClipImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    ClipMetalParam params{min_, max_};
    params_buffer_ = std::make_shared<MetalBuffer>(metal_context_, sizeof(params), &params);
    function_name_ = "clip";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

ClipImageCompute::~ClipImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(clip,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ClipImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(clip,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ClipImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
