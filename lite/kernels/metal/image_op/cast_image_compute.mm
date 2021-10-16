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

#include "lite/kernels/metal/image_op/cast_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void CastImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
    auto input_dims = param.X->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void CastImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void CastImageCompute::run_without_mps() {
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

void CastImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    auto valid = true;
    // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
    if (param.in_dtype == 3 || param.in_dtype == 6 || param.out_dtype == 3 ||
        param.out_dtype == 6) {
        valid = false;
    }
    if (!valid) {
        LOG(FATAL) << "cast from in_type(" << param.in_dtype << ") to out_type(" << param.out_dtype
                   << ") is not supported.";
    }

    CastMetalParam metal_params = {(int)param.in_dtype, (int)param.out_dtype};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_params), &metal_params);

    // pipline
    function_name_ = "cast";
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

CastImageCompute::~CastImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(cast,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::CastImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kAny), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::CastImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kAny), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
