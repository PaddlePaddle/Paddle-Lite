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

#include "lite/kernels/metal/image_op/concat_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void ConcatImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.output->dims();
    int num = (int)param.x.size();
#ifdef LITE_WITH_METAL_FULL

#else
    for (int i = 0; i < num; i++) {
        auto input_image = param.x[i]->data<MetalHalf, MetalImage>();
        input_buffers_.emplace_back(input_image);
    }
    output_buffer_ = param.output->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void ConcatImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void ConcatImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    int idx = 0;
    auto encoder = [backend commandEncoder];
    [encoder setTexture:output_buffer_->image() atIndex:(idx++)];
    if (v_ == "normal") {
        [encoder setTexture:output_buffer_->image() atIndex:(idx++)];
    }
    for (auto item : input_buffers_) {
        [encoder setTexture:item->image() atIndex:(idx++)];
    }

    [encoder setBuffer:params_buffer_->buffer() offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void ConcatImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    int num = (int)param.x.size();
    int vaxis = 0;
    int axis = int(4 - output_buffer_->tensor_dim_.size() + param.axis);
    auto* axis_tensor = param.axis_tensor;
    if (axis_tensor != nullptr) {
        auto* axis_tensor_data = axis_tensor->data<int>();
        axis = axis_tensor_data[0];
    }
    if (axis < 0) {
        axis += output_buffer_->tensor_dim_.size();
    }
    auto orank = output_buffer_->tensor_dim_.size();
    assert(num <= 6);

    std::vector<int> transpose = {0, 2, 3, 1};
    if (output_buffer_->tensor_dim_.size() < 4) {
        transpose = {0, 1, 2, 3};
    }
    for (int i = 0; i < 4; i++) {
        if (transpose[i] == axis) {
            axis = i;
            break;
        }
    }

    int vdim[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < num; i++) {
        vdim[i] = (int)input_buffers_[i]->dim_[axis];
    }
    if (axis == 1) {
        v_ = "y";
    } else if (axis == 2) {
        v_ = "x";
    } else {
        if ((output_buffer_->dim_[0] == 1) && (axis == 3)) {
            auto vz = true;
            for (int i = 0; i < num; i++) {
                if (vdim[i] % 4 != 0) {
                    vz = false;
                    break;
                }
            }
            if (vz) {
                v_ = "z";
                for (int i = 0; i < num; i++) {
                    vdim[i] = vdim[i] / 4;
                }
            }
        }
    }

    std::vector<int> odm = {1, 1, 1, 1};
    if (output_buffer_->tensor_dim_.size() == 4) {
        for (int i = 0; i < orank; i++) {
            odm[i] = (int)(output_buffer_->dim_[i]);
        }
    } else {
        for (int i = 0; i < orank; i++) {
            odm[4 - orank + i] = (int)(output_buffer_->tensor_dim_[i]);
        }
    }

    if (v_ == "normal")
        vaxis = 1;
    else if (v_ == "x")
        vaxis = 2;
    else if (v_ == "y")
        vaxis = 3;
    else if (v_ == "z")
        vaxis = 4;

    ConcatMetalParam concat_params{{odm[0], odm[1], odm[2], odm[3]},
        static_cast<int>(axis),
        0,
        num,
        vaxis,
        {transpose[0], transpose[1], transpose[2], transpose[3]},
        {(int)vdim[0], (int)vdim[1], (int)vdim[2], (int)vdim[3], (int)vdim[4], (int)vdim[5]}};

    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(concat_params), &concat_params);
#ifdef LITE_WITH_METAL_FULL
#else
    if (v_ == "normal")
        function_name_ = "concat_normal";
    else
        function_name_ = "concat";
#endif
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

ConcatImageCompute::~ConcatImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ConcatImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("AxisTensor",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(concat,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ConcatImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("AxisTensor",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
