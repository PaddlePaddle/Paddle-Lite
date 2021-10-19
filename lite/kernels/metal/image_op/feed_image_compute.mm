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

#include "lite/kernels/metal/image_op/feed_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/backends/metal/metal_mtl_data.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void FeedImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    init_memory();
    setup_pipeline();
}

void FeedImageCompute::ReInitWhenNeeded() {
    const auto& param = this->Param<param_t>();
    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();

    if (last_input_dims_ != input_dims) {
        release_memory();
        init_memory();
    }
}

void FeedImageCompute::init_memory() {
    const auto& param = this->Param<param_t>();
    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();

    param.out->Resize(input_dims);
#ifdef LITE_WITH_METAL_FULL
#else
    output_buffer_ = param.out->mutable_data<MetalHalf, MetalImage>(metal_context_, input_dims);
#endif
    last_input_dims_ = input_dims;
}

void FeedImageCompute::Run() {
    @autoreleasepool {
        const auto& param = this->Param<param_t>();
        Tensor& input_tensor = param.feed_list->at(param.col);
        if (input_tensor.metal_data_type() == Tensor::MetalDataType::kRaw) {
            run_raw();
        } else if (input_tensor.metal_data_type() == Tensor::MetalDataType::kMetal) {
            run_mtl_texture();
        }
    }
}

void FeedImageCompute::setup_pipeline() {
    const auto& param = this->Param<param_t>();
    Tensor& input_tensor = param.feed_list->at(param.col);
    if (input_tensor.metal_data_type() == Tensor::MetalDataType::kRaw) {
        if (input_tensor.precision() == lite_api::PrecisionType::kFloat) {
            setup_float();
        }
    } else if (input_tensor.metal_data_type() == Tensor::MetalDataType::kMetal) {
        setup_mtl_texture();
    }
}

#pragma mark - float&int32

void FeedImageCompute::run_raw() {
    const auto& param = this->Param<param_t>();
    Tensor& input_tensor = param.feed_list->at(param.col);

    if (input_tensor.precision() == lite_api::PrecisionType::kFloat) {
        run_float();
    } else if (input_tensor.precision() == lite_api::PrecisionType::kInt32) {
        run_int32();
    }
}

void FeedImageCompute::run_float() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    const auto& param = this->Param<param_t>();
    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_buffer = input_tensor.data<float>();
    auto input_dims = input_tensor.dims();
    int mem_size = (int)input_dims.production() * sizeof(float);
    input_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, mem_size, const_cast<float*>(input_buffer));

    auto encoder = [backend commandEncoder];
    [encoder setBuffer:input_buffer_->buffer() offset:(0) atIndex:(0)];
    [encoder setTexture:(output_buffer_->image()) atIndex:(0)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(1)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void FeedImageCompute::setup_float() {
    const auto& param = this->Param<param_t>();

    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();
    auto irank = input_dims.size();
    std::vector<int> idm = {1, 1, 1, 1};
    for (int i = 0; i < irank; i++) {
        idm[4 - irank + i] = (int)(input_dims[i]);
    }
    FeedMetalParam metal_params{(int)irank, {idm[0], idm[1], idm[2], idm[3]}};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_params), &metal_params);

    function_name_ = "buf_to_tex_c_n";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

void FeedImageCompute::run_int32() {
    const auto& param = this->Param<param_t>();

    auto output_int32_ = param.out->mutable_data<int32_t>();

    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();
    auto input_int32 = input_tensor.data<int32_t>();

    auto len = input_dims.production() * sizeof(int32_t);
    memcpy(output_int32_, input_int32, len);
}

#pragma mark - texture

void FeedImageCompute::run_mtl_texture() {
    const auto& param = this->Param<param_t>();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();
    if (input_dims.size() != 4) {
        return;
    }
    // need scale
    do {
        id<MTLTexture> inTexture = input_tensor.data<MetalMTLData>()->image();
        auto cmdbuf = [backend commandBuffer];
        [(__bridge MPSImageLanczosScale*)lanczos_ encodeToCommandBuffer:cmdbuf
                                                          sourceTexture:inTexture
                                                     destinationTexture:resize_texture_];
        [backend commit:cmdbuf];
    } while (0);

    // texture to texture_array
    do {
        auto pipline = pipline_;
        auto outTexture = output_buffer_->image();

        auto encoder = [backend commandEncoder];
        [encoder setTexture:resize_texture_ atIndex:(0)];
        [encoder setTexture:(output_buffer_->image()) atIndex:(1)];

        [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
        [backend commit];
    } while (0);
}

void FeedImageCompute::setup_mtl_texture() {
    const auto& param = this->Param<param_t>();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();
    NSMutableArray* dimsAry = [NSMutableArray arrayWithCapacity:3];
    for (int i = 0; i < input_dims.size(); i++) {
        [dimsAry addObject:@(input_dims[i])];
    }
    resize_texture_ = [backend lanczosTextureCreate:dimsAry];
    lanczos_ = (__bridge_retained void*)[backend lanczosScalePtrCreate];
    function_name_ = "texture2d_to_2d_array";
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - internal

void FeedImageCompute::release_memory() {
    if (lanczos_) {
        CFRelease(lanczos_);
        lanczos_ = nullptr;
    }
    if (resize_texture_) {
        resize_texture_ = nil;
    }
    TargetWrapperMetal::FreeImage(output_buffer_);
}

FeedImageCompute::~FeedImageCompute() {
    release_memory();
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(feed,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::FeedImageCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(feed,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::FeedImageCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(feed, kMetal, kAny, kAny, paddle::lite::kernels::metal::FeedImageCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kAny), DATALAYOUT(kAny))})
    .Finalize();
