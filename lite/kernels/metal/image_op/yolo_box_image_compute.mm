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

#include "lite/kernels/metal/image_op/yolo_box_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void YoloBoxImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->template Param<param_t>();
    auto output_boxes = param.Boxes->dims();
    auto output_scores = param.Scores->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_x_ = param.X->data<MetalHalf, MetalImage>();
    input_imgSize_ = param.ImgSize->data<int32_t>();
    output_boxes_ = param.Boxes->mutable_data<MetalHalf, MetalImage>(metal_context_, output_boxes);
    output_scores_ =
        param.Scores->mutable_data<MetalHalf, MetalImage>(metal_context_, output_scores);
#endif
    setup_without_mps();
}

void YoloBoxImageCompute::Run() {
    @autoreleasepool {
        reset_data();
        run_tex_to_buf();
        run_yolo_box();
        run_buf_to_tex_boxes();
        run_buf_to_tex_scores();
    }
}

void YoloBoxImageCompute::reset_data() {
    TargetWrapperMetal::MemsetSync(intermediate_input_x_.contents, 0, intermediate_input_x_.length);
    TargetWrapperMetal::MemsetSync(intermediate_boxes_.contents, 0, intermediate_boxes_.length);
    TargetWrapperMetal::MemsetSync(intermediate_scores_.contents, 0, intermediate_scores_.length);
}

void YoloBoxImageCompute::run_tex_to_buf() {
    auto pipline = pipline_tex_to_buf;
    auto outTexture = input_buffer_x_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_x_->image() atIndex:(0)];
    [encoder setBuffer:intermediate_input_x_ offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void YoloBoxImageCompute::run_yolo_box() {
    const auto& param = this->Param<param_t>();
    auto pipline = pipline_;
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setBuffer:intermediate_input_x_ offset:(0) atIndex:(0)];
    [encoder setBuffer:anchors_buffer_ offset:(0) atIndex:(1)];
    [encoder setBuffer:intermediate_boxes_ offset:(0) atIndex:(2)];
    [encoder setBuffer:intermediate_scores_ offset:(0) atIndex:(3)];
    [encoder setBuffer:params_buffer_->buffer() offset:(0) atIndex:(4)];


    auto N = param.anchors.size() / 2;
    auto H = input_buffer_x_->pad_to_four_dim_[2];
    auto W = input_buffer_x_->pad_to_four_dim_[3];

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

void YoloBoxImageCompute::run_buf_to_tex_boxes() {
    auto pipline = pipline_buf_to_tex;
    auto outTexture = output_boxes_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setBuffer:intermediate_boxes_ offset:(0) atIndex:(0)];
    [encoder setTexture:output_boxes_->image() atIndex:(0)];
    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void YoloBoxImageCompute::run_buf_to_tex_scores() {
    auto pipline = pipline_buf_to_tex;
    auto outTexture = output_scores_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setBuffer:intermediate_scores_ offset:(0) atIndex:(0)];
    [encoder setTexture:output_scores_->image() atIndex:(0)];
    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void YoloBoxImageCompute::setup_without_mps() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    const auto& param = this->Param<param_t>();

    auto xN = input_buffer_x_->pad_to_four_dim_[0];
    auto xC = input_buffer_x_->pad_to_four_dim_[1];
    auto xH = input_buffer_x_->pad_to_four_dim_[2];
    auto xW = input_buffer_x_->pad_to_four_dim_[3];
    auto xSize = param.downsample_ratio * xH;

    auto imgH = xSize;
    auto imgW = xSize;
    if (input_imgSize_) {
        imgH = input_imgSize_[0];
        imgW = input_imgSize_[1];
    }

    auto anchorsNumber = param.anchors.size() / 2;
    auto xStride = xH * xW;
    auto anchorsStride = (param.class_num + 5) * xStride;

    YoloBoxMetalParam metal_params = {(int)imgH,
        (int)imgW,
        (int)xN,
        (int)xC,
        (int)xH,
        (int)xW,
        (int)xStride,
        (int)xSize,
        (int)(output_boxes_->pad_to_four_dim_[2]),
        (int)anchorsNumber,
        (int)anchorsStride,
        (int)(param.class_num),
        (int)(param.clip_bbox ? 1 : 0),
        (float)param.conf_thresh,
        (float)param.scale_x_y,
        (float)(-0.5 * (param.scale_x_y - 1.0))};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_params), &metal_params);

    auto anchorLength = sizeof(int) * param.anchors.size();
    void* anchorData = (void*)param.anchors.data();
    anchors_buffer_ = [backend newDeviceBuffer:anchorLength
                                         bytes:anchorData
                                        access:METAL_ACCESS_FLAG::CPUWriteOnly];
    auto inputXLength = input_buffer_x_->dim_.production() * sizeof(MetalHalf);
    intermediate_input_x_ =
        [backend newDeviceBuffer:inputXLength access:METAL_ACCESS_FLAG::CPUWriteOnly];

    auto outputBoxesLength = output_boxes_->dim_.production() * sizeof(MetalHalf);
    intermediate_boxes_ =
        [backend newDeviceBuffer:outputBoxesLength access:METAL_ACCESS_FLAG::CPUWriteOnly];

    auto outputScoresLength = output_scores_->dim_.production() * sizeof(MetalHalf);
    intermediate_scores_ =
        [backend newDeviceBuffer:outputScoresLength access:METAL_ACCESS_FLAG::CPUWriteOnly];

    // pipline
    function_name_ = "yolo_box";
    pipline_ = [backend pipline:function_name_];

    pipline_tex_to_buf = [backend pipline:"tex2d_ary_to_buf"];
    pipline_buf_to_tex = [backend pipline:"buf_h_to_tex_h"];
}

YoloBoxImageCompute::~YoloBoxImageCompute() {
    TargetWrapperMetal::FreeImage(output_boxes_);
    TargetWrapperMetal::FreeImage(output_scores_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::YoloBoxImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("ImgSize",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Boxes",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Scores",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(yolo_box,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::YoloBoxImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("ImgSize",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Boxes",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Scores",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
