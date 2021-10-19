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

#include "lite/kernels/metal/image_op/conv2d_transpose_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_converter.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/backends/metal/mps_conv_datasource.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {
void Conv2dTransposeImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    init_memory();
    init_for_run();
}

void Conv2dTransposeImageCompute::init_memory() {
    const auto& param = this->Param<param_t>();
    auto input_dims = param.x->dims();
    auto filter_dims = param.filter->dims();
    auto output_dims = param.output->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.x->data<MetalHalf, MetalImage>();
    output_buffer_ = param.output->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
    if (param.bias) {
        bias_buffer_ = param.bias->data<MetalHalf, MetalImage>();
    } else {
        auto* blank_host = (float*)TargetWrapperMetal::Malloc(sizeof(float) * output_dims[1]);
        TargetWrapperMetal::MemsetSync(blank_host, 0, sizeof(MetalHalf) * output_dims[1]);
        DDim blank_dim = DDimLite({output_dims[1]});
        Tensor blank_tensor_;
        blank_tensor_.Resize(blank_dim);
        blank_buffer_ =
            blank_tensor_.mutable_data<MetalHalf, MetalImage>(metal_context_, blank_dim);
        blank_buffer_->CopyFromNCHW<float>(blank_host);
        TargetWrapperMetal::Free(blank_host);
        blank_host = nullptr;
    }
#endif
    last_input_dims_ = input_dims;
}

void Conv2dTransposeImageCompute::init_for_run() {
    const auto& param = this->Param<param_t>();

    function_name_ = KernelFunctionName(param);

    // use mps or not
    bool should_use_mps = false;
    if (@available(iOS 11.3, *)) {
        if (metal_context_->use_mps()) {
            // TODO Daming6432 mps support
        }
    }

    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        if (function_name_.empty()) {
            LOG(FATAL) << "conv2d: cannot find the name";
        } else {
            setup_without_mps();
        }
    }
}
void Conv2dTransposeImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    auto padLeft = (*param.paddings)[2];
    auto padTop = (*param.paddings)[0];
    assert((*param.paddings)[0] == (*param.paddings)[1]);

    auto filterWidth = param.filter->dims()[3];
    auto filterHeight = param.filter->dims()[2];

    auto kernelWidth = (uint16_t)(filterWidth);
    auto kernelHeight = (uint16_t)(filterHeight);

    auto strideX = uint16_t(param.strides[1]);
    auto strideY = uint16_t(param.strides[0]);
    auto paddingX = uint16_t((*param.paddings)[2]);
    auto paddingY = uint16_t((*param.paddings)[0]);
    auto dilationX = uint16_t((*param.dilations)[1]);
    auto dilationY = uint16_t((*param.dilations)[0]);
    auto hasAdd = (uint16_t)((param.bias) ? 1 : 0);
    auto groups = uint16_t(param.groups);
    auto inputC = uint16_t(input_buffer_->tensor_dim_[1]);
    auto filterC = uint16_t(param.filter->dims()[1]);
    auto outputC = uint16_t(param.output->dims()[1]);

    // add
    int xdim[4], ydim[4], xtrans[4], ytrans[4];
    for (int i = 0; i < 4; i++) {
        xdim[i] = (int)output_buffer_->dim_[i];
        ydim[i] = (int)bias_buffer_->dim_[i];
    }

    int axis = -1;
    int params_axis;
    if (axis == -1) {
        params_axis = 4 - (int)(output_buffer_->tensor_dim_.size());
    } else {
        params_axis = 4 - (int)(output_buffer_->tensor_dim_.size()) + axis;
    }

    int params_fast = 0;
    if ((output_buffer_->dim_ == bias_buffer_->dim_) &&
        (output_buffer_->transpose_ == bias_buffer_->transpose_)) {
        //      print("===> elementwise_add fast!!!")
        params_fast = 1;
    }

    int add_by_channel = 0;
    if (bias_buffer_->tensor_dim_.size() == 1 &&
        (axis == 1 ||
            (axis == -1 && bias_buffer_->tensor_dim_[0] == output_buffer_->pad_to_four_dim_[1]))) {
        add_by_channel = 1;
    }

    ElementwiseAddMetalParam addParam = {params_fast,
        add_by_channel,
        params_axis,
        (int)output_buffer_->tensor_dim_.size(),
        {xdim[0], xdim[1], xdim[2], xdim[3]},
        {output_buffer_->transpose_[0],
            output_buffer_->transpose_[1],
            output_buffer_->transpose_[2],
            output_buffer_->transpose_[3]},
        {ydim[0], ydim[1], ydim[2], ydim[3]},
        {bias_buffer_->transpose_[0],
            bias_buffer_->transpose_[1],
            bias_buffer_->transpose_[2],
            bias_buffer_->transpose_[3]}};
    ConvTransposeAddMetalParam metalParam = {kernelWidth,
        kernelHeight,
        strideX,
        strideY,
        paddingX,
        paddingY,
        dilationX,
        dilationY,
        groups,
        inputC,
        filterC,
        outputC,
        hasAdd,
        addParam};

    params_buffer_ = std::make_shared<MetalBuffer>(metal_context_, sizeof(metalParam), &metalParam);

    if (HasPrefix(function_name_, "conv_transpose2x2")) {
        filter_buffer_ = std::make_shared<MetalBuffer>(metal_context_, param.filter->dims());
    } else {
        throw std::logic_error("ERROR: conv_transpose still cannot support this");
    }
    auto filter_buffer = param.filter->data<float>();
    filter_buffer_->CopyFromNCHW<float>(filter_buffer);

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

void Conv2dTransposeImageCompute::ReInitWhenNeeded() {
    const auto& param = this->Param<param_t>();
    auto input_dims = param.x->dims();

    if (last_input_dims_ != input_dims) {
        release_memory();
        init_memory();

        if (use_mps_) {
            // TODO daming5432
        }
    }
}
void Conv2dTransposeImageCompute::release_memory() {
    TargetWrapperMetal::FreeImage(output_buffer_);
    TargetWrapperMetal::FreeImage(blank_buffer_);
}

void Conv2dTransposeImageCompute::Run() {
    if (use_mps_) {
        run_with_mps();
    } else {
        run_without_mps();
    }
}

void Conv2dTransposeImageCompute::run_without_mps() {
    const auto& param = this->Param<param_t>();
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    if (param.bias) {
        [encoder setTexture:(bias_buffer_->image()) atIndex:(1)];
    } else {
        [encoder setTexture:(blank_buffer_->image()) atIndex:(1)];
    }
    [encoder setTexture:(output_buffer_->image()) atIndex:(2)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];
    [encoder setBuffer:(filter_buffer_->buffer()) offset:(0) atIndex:(1)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void Conv2dTransposeImageCompute::setup_with_mps() {
    // TODO daming5432
}
void Conv2dTransposeImageCompute::run_with_mps() {
    // TODO daming5432
}

std::string Conv2dTransposeImageCompute::KernelFunctionName(const param_t& param) {
    if (param.filter->dims()[3] == 2 && param.filter->dims()[2] == 2) {
        if (param.strides[0] == 2 && param.strides[1] == 2) {
            return "conv_transpose2x2_stride2";
        }
    } else if (param.filter->dims()[3] == 3 && param.filter->dims()[2] == 3) {
        if (param.strides[0] == 2 && param.strides[1] == 2) {
            return "conv_transpose3x3_stride2";
        }
    }
    return "";
}

bool Conv2dTransposeImageCompute::HasPrefix(const std::string& function_name,
    const std::string& prefix) {
    if (function_name.size() >= prefix.size() &&
        function_name.compare(0, prefix.size(), prefix) == 0) {
        return true;
    }
    return false;
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d_transpose,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::Conv2dTransposeImageCompute,
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

REGISTER_LITE_KERNEL(conv2d_transpose,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::Conv2dTransposeImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Filter",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP16), DATALAYOUT(kNCHW))})
    .BindOutput("Output",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
