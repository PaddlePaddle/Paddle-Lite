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

#include "lite/kernels/metal/image_op/elementwise_div_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void ElementwiseDivImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
    auto input_dims = param.X->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_x_ = param.X->data<MetalHalf, MetalImage>();
    input_buffer_y_ = param.Y->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void ElementwiseDivImageCompute::Run() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_x_->image() atIndex:(0)];
    [encoder setTexture:input_buffer_y_->image() atIndex:(1)];
    [encoder setTexture:output_buffer_->image() atIndex:(2)];
    [encoder setBuffer:params_buffer_->buffer() offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void ElementwiseDivImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    bool valid = false;
    int by_channel = 0;
    if (input_buffer_x_->tensor_dim_.size() == 4) {
        if (input_buffer_y_->tensor_dim_.size() == 4) {
            if (input_buffer_y_->tensor_dim_[0] == 1 &&
                input_buffer_y_->tensor_dim_[2] == 1 &&
                input_buffer_y_->tensor_dim_[3] == 1 &&
                input_buffer_x_->tensor_dim_[1] == input_buffer_y_->tensor_dim_[1]) {
                by_channel = 1;
            } else {
                for (int i = 0; i < 4; i++) {
                    if (input_buffer_x_->tensor_dim_[i] != input_buffer_y_->tensor_dim_[i]) {
                        valid = false;
                        break;
                    }
                }
            }
        } else if (input_buffer_y_->tensor_dim_.size() == 3) {
            if (param.axis == 1 || param.axis == -1) {
                if (input_buffer_y_->tensor_dim_[1] == 1 && input_buffer_y_->tensor_dim_[2] == 1 &&
                    input_buffer_y_->tensor_dim_[0] == input_buffer_x_->tensor_dim_[1]) {
                    by_channel = 1;
                }
            }
        } else if (input_buffer_y_->tensor_dim_.size() == 2) {
            if (param.axis == 0 || param.axis == -1) {
                by_channel = 1;
            }
        } else if (input_buffer_y_->tensor_dim_.size() == 1) {
            by_channel = 1;
        } else {
            valid = false;
        }
    } else {
        valid = false;
    }
    if (!valid) {
        LOG(FATAL) << "elementwise_div: only supports : 1.same shapes 2.by channel.";
    }

    ElementwiseMetalParam element_params = {by_channel};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(element_params), &element_params);

    // input source: 4 dims-from last tensor, 3 dims-from out tensor
    function_name_ = "elementwise_div";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

ElementwiseDivImageCompute::~ElementwiseDivImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_div,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseDivImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseDivImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
