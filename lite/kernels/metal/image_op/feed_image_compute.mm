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

#include "lite/kernels/metal/image_op/feed_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void FeedImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.out->dims();

    Tensor& input_tensor = param.feed_list->at(param.col);
    auto input_dims = input_tensor.dims();
    param.out->Resize(input_dims);
#ifdef LITE_WITH_METAL_FULL
#else
    output_buffer_ = param.out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif
    auto input_c = input_dims[1];
    if (input_c == 1) {
        function_name_ = "buf_to_tex";
    } else if (input_c == 3) {
        function_name_ = "buf_to_tex_c_3";
    } else {
        function_name_ = "buf_to_tex_c_n";
    }
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

void FeedImageCompute::Run() {
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

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

FeedImageCompute::~FeedImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
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
