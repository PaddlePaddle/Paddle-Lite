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

#include "lite/kernels/metal/image_op/slice_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void SliceImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);

#endif
    setup_without_mps();
}

void SliceImageCompute::Run() {
    @autoreleasepool {
        reset_data();
        run_tex_to_buf();
        run_without_mps();
        run_buf_to_tex();
    }
}

void SliceImageCompute::reset_data() {
    TargetWrapperMetal::MemsetSync(intermediate_input_.contents, 0, intermediate_input_.length);
    TargetWrapperMetal::MemsetSync(intermediate_output_.contents, 0, intermediate_output_.length);
}

void SliceImageCompute::run_tex_to_buf() {
    auto pipline = pipline_tex_to_buf;
    auto outTexture = input_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_->image() atIndex:(0)];
    [encoder setBuffer:intermediate_input_ offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void SliceImageCompute::run_buf_to_tex() {
    auto pipline = pipline_buf_to_tex;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setBuffer:intermediate_output_ offset:(0) atIndex:(0)];
    [encoder setTexture:output_buffer_->image() atIndex:(0)];
    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void SliceImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setBuffer:intermediate_input_ offset:(0) atIndex:(0)];
    [encoder setBuffer:intermediate_output_ offset:(0) atIndex:(1)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(2)];

    auto N = input_buffer_->pad_to_four_dim_[0];
    auto C = input_buffer_->pad_to_four_dim_[1];
    auto H = input_buffer_->pad_to_four_dim_[2];
    auto W = input_buffer_->pad_to_four_dim_[3];

    auto slices = (N + 3) / 4;

    auto width = MIN(W, pipline.threadExecutionWidth);
    auto height = MIN(H, pipline.maxTotalThreadsPerThreadgroup / width);
    auto threadsPerGroup = MTLSizeMake(width, height, 1);

    auto groupWidth = (W + width - 1) / width;
    auto groupHeight = (H + height - 1) / height;
    auto groups = MTLSizeMake(groupWidth, groupHeight, N ? N : slices);

    [backend dispatchEncoder:encoder pipline:pipline threadsPerGroup:threadsPerGroup groups:groups];
    [backend commit];
}

void SliceImageCompute::setup_without_mps() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    const auto& param = this->Param<param_t>();
    const auto in_dims = input_buffer_->pad_to_four_dim_;
    const auto out_dims = output_buffer_->pad_to_four_dim_;
    const auto in_tensor_dims = input_buffer_->tensor_dim_;
    const auto out_tensor_dims = output_buffer_->tensor_dim_;

    auto axes = param.axes;
    auto starts = param.starts;
    auto ends = param.ends;
    std::vector<int> real_starts(in_tensor_dims.size(), 0);
    std::vector<int> real_ends(in_tensor_dims.size(), 0);
    for (int i = 0; i < axes.size(); i++) {
        int dim_value = in_tensor_dims[axes[i]];
        if (dim_value > 0) {
            int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
            int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
            start = std::max(start, 0);
            end = std::max(end, 0);
            end = std::min(end, dim_value);
            real_starts[axes[i]] = start;
            real_ends[axes[i]] = end;
        }
    }
    for (int i = out_tensor_dims.size(); i < 4; i++) {
        real_starts.insert(real_starts.begin(), 0);
        real_ends.insert(real_ends.begin(), 0);
    }
    real_ends[1] = real_ends[1] == 0 ? in_dims[1] : real_ends[1];
    SliceMetalParam params{(int)in_dims[3],
        (int)in_dims[2],
        (int)out_dims[3],
        (int)out_dims[2],
        (int)(in_dims[3] * in_dims[2]),
        (int)(out_dims[3] * out_dims[2]),
        ((int)out_dims[1] + 3) / 4,
        {real_starts[0], real_starts[1], real_starts[2], real_starts[3]},
        real_ends[1]};

    params_buffer_ = std::make_shared<MetalBuffer>(metal_context_, sizeof(params), &params);

    auto inputLength = input_buffer_->dim_.production() * sizeof(MetalHalf);
    intermediate_input_ =
        [backend newDeviceBuffer:inputLength access:METAL_ACCESS_FLAG::CPUWriteOnly];
    auto outputLength = output_buffer_->dim_.production() * sizeof(MetalHalf);
    intermediate_output_ =
        [backend newDeviceBuffer:outputLength access:METAL_ACCESS_FLAG::CPUWriteOnly];

    function_name_ = "slice";
    // pipline
    pipline_ = [backend pipline:function_name_];
    pipline_tex_to_buf = [backend pipline:"tex2d_ary_to_buf"];
    pipline_buf_to_tex = [backend pipline:"buf_h_to_tex_h"];
}

SliceImageCompute::~SliceImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(slice,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SliceImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(slice,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SliceImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
