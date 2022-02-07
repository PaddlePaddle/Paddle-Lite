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

#include "lite/kernels/metal/image_op/split_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void SplitImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.x->data<MetalHalf, MetalImage>();

    auto outputs = param.output;
    size_t num = outputs.size();
    for (int i = 0; i < num; i++) {
        auto output_dims = outputs[i]->dims();
        auto output_image =
            outputs[i]->template mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
        output_buffers_.emplace_back(output_image);
    }
#endif

    setup_without_mps();
}

void SplitImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void SplitImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = input_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    NSUInteger index = 1;
    for (auto item : output_buffers_) {
        [encoder setTexture:(item->image()) atIndex:(index++)];
    }
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    if (split_v_ != "zz") {
        [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    } else {
        NSUInteger z = 0;
        z += (metal_param_.vdim[0] + 3) / 4 * 4;
        z += (metal_param_.vdim[1] + 3) / 4 * 4;
        z += (metal_param_.vdim[2] + 3) / 4 * 4;
        z += (metal_param_.vdim[3] + 3) / 4 * 4;

        [backend dispatchEncoder:encoder
                         pipline:pipline
                    threadsShape:@[ @(z), @(outTexture.height), @(outTexture.width) ]];
    }
    [backend commit];
}

void SplitImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    auto outputs = param.output;
    int num = outputs.size();
    int vaxis = 0;
    int irank = (int)input_buffer_->tensor_dim_.size();

    // intput dims: CPU NCHW
    std::vector<int> idm = {1, 1, 1, 1};
    for (int i = 0; i < irank; i++) {
        idm[4 - irank + i] = (int)input_buffer_->tensor_dim_[i];
    }

    int axis = int(4 - input_buffer_->tensor_dim_.size() + param.axis);
    auto* axis_tensor = param.axis_tensor;
    if (axis_tensor != nullptr) {
        auto* axis_tensor_data = axis_tensor->data<int>();
        axis = axis_tensor_data[0];
    }
    if (axis < 0) {
        axis += input_buffer_->tensor_dim_.size();
    }

    std::vector<int> trans = {0, 2, 3, 1};
    if (input_buffer_->tensor_dim_.size() < 4) {
        trans = {0, 1, 2, 3};
    }
    for (int i = 0; i < 4; i++) {
        if (trans[i] == axis) {
            axis = i;
            break;
        }
    }

    int vdim[4]{0, 0, 0, 0};
    for (int i = 0; i < num; i++) {
        vdim[i] = int(param.output[i]->dims()[param.axis]);
    }
    std::string v_ = "normal";

    if (axis == 1) {
        v_ = "y";
    } else if (axis == 2) {
        v_ = "x";
    } else if (axis == 3) {
        auto vz = true;
        for (int i = 0; i < num; i++) {
            if (vdim[i] % 4 != 0) {
                vz = false;
                break;
            }
        }
        if (vz) {
            v_ = "z";
            vdim[0] = vdim[0] / 4;
            vdim[1] = vdim[1] / 4;
            vdim[2] = vdim[2] / 4;
            vdim[3] = vdim[3] / 4;
        } else {
            v_ = "zz";
        }
    }

    if (v_ == "normal") {
        throw std::logic_error("ERROR: unsupported split type");
    }

    if (v_ == "normal")
        vaxis = 0;
    else if (v_ == "x")
        vaxis = 1;
    else if (v_ == "y")
        vaxis = 2;
    else if (v_ == "z")
        vaxis = 3;
    else if (v_ == "zz")
        vaxis = 4;
    split_v_ = v_;

    SplitMetalParam metal_param = {{idm[0], idm[1], idm[2], idm[3]},
        static_cast<int>(axis),
        0,
        num,
        vaxis,
        {trans[0], trans[1], trans[2], trans[3]},
        {(int)vdim[0], (int)vdim[1], (int)vdim[2], (int)vdim[3]}};
    metal_param_ = metal_param;

    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_param), &metal_param);

    if (v_ == "zz")
        function_name_ = "split_zz";
    else
        function_name_ = "split";

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

SplitImageCompute::~SplitImageCompute() {
    for (auto item : output_buffers_) {
        TargetWrapperMetal::FreeImage(item);
    }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(split,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SplitImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("AxisTensor",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SectionsTensorList",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(split,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SplitImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("AxisTensor",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SectionsTensorList",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
