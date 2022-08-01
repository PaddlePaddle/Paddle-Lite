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
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void Conv2dTransposeImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    init_attention();
    init_memory();
    init_for_run();
}

void Conv2dTransposeImageCompute::ReInitWhenNeeded() {
    const auto& param = this->Param<param_t>();
    auto input_dims = param.x->dims();

    if (last_input_dims_ != input_dims) {
        release_memory();
        init_memory();

        if (use_mps_) {
            if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
                if (mps_input_image_) {
                    CFRelease(mps_input_image_);
                    mps_input_image_ = nullptr;
                }
                if (mps_output_image_) {
                    CFRelease(mps_output_image_);
                    mps_output_image_ = nullptr;
                }
                auto input_c = static_cast<int>(input_buffer_->tensor_dim_[1]);
                auto output_c = static_cast<int>(output_buffer_->tensor_dim_[1]);
                // MPS input and output
                mps_input_image_ = (__bridge_retained void*)[[MPSImage alloc]
                    initWithTexture:input_buffer_->image()
                    featureChannels:input_c];
                mps_output_image_ = (__bridge_retained void*)[[MPSImage alloc]
                    initWithTexture:output_buffer_->image()
                    featureChannels:output_c];
            }
        }
    }
}

// attention!!! filter: CNHW2NCHW
void Conv2dTransposeImageCompute::init_attention() {
    const auto& param = this->Param<param_t>();
    auto dims = param.filter->dims();
    auto dims_nchw = DDimLite({dims[1], dims[0], dims[2], dims[3]});
    filter_metal_dims_ = dims_nchw;
}

void Conv2dTransposeImageCompute::init_memory() {
    const auto& param = this->Param<param_t>();

    auto input_dims = param.x->dims();
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
    bool should_use_mps = false;
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        if (metal_context_->use_mps()) {
            should_use_mps = true;
        }
    }
    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        if (function_name_.empty()) {
            LOG(FATAL) << "conv2d_transpose: cannot find the name";
        } else {
            setup_without_mps();
        }
    }
}

void Conv2dTransposeImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF

void Conv2dTransposeImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    auto filterWidth = filter_metal_dims_[3];
    auto filterHeight = filter_metal_dims_[2];
    auto kernelWidth = (uint16_t)(filterWidth);
    auto kernelHeight = (uint16_t)(filterHeight);

    auto strideX = uint16_t(param.strides[1]);
    auto strideY = uint16_t(param.strides[0]);
    auto paddingX = uint16_t((*param.paddings)[2]);
    auto paddingY = uint16_t((*param.paddings)[0]);
    auto dilationX = uint16_t((*param.dilations)[1]);
    auto dilationY = uint16_t((*param.dilations)[0]);

    auto groups = uint16_t(param.groups);
    auto inputC = uint16_t(input_buffer_->tensor_dim_[1]);
    auto filterC = uint16_t(filter_metal_dims_[1]);
    auto outputC = uint16_t(param.output->dims()[1]);

    auto hasAdd = (uint16_t)((param.bias) ? 1 : 0);

    // add
    ElementwiseAddMetalParam element_params = {};
    if (param.bias) {
        int xdim[4], ydim[4];
        for (int i = 0; i < 4; i++) {
            xdim[i] = (int)output_buffer_->dim_[i];
            ydim[i] = (int)bias_buffer_->dim_[i];
        }

        int axis = -1;
        int params_axis;
        if (axis == -1) {
            params_axis = 4 - (int)(bias_buffer_->tensor_dim_.size());
        } else {
            params_axis = 4 - (int)(output_buffer_->tensor_dim_.size()) + axis;
        }

        int params_fast = 0;
        if ((output_buffer_->dim_ == bias_buffer_->dim_) &&
            (output_buffer_->transpose_ == bias_buffer_->transpose_)) {
            params_fast = 1;
        }

        int add_by_channel = 0;
        if (bias_buffer_->tensor_dim_.size() == 1 &&
            (axis == 1 ||
                (axis == -1 &&
                    bias_buffer_->tensor_dim_[0] == output_buffer_->pad_to_four_dim_[1]))) {
            add_by_channel = 1;
        }
        if (add_by_channel == 1 || params_fast == 1) {
        } else {
            LOG(FATAL) << "conv2d: add only support by channel";
        }

        element_params = {params_fast,
            add_by_channel,
            params_axis,
            (int)bias_buffer_->tensor_dim_.size(),
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
    } else {
    }

    // activate
    uint16_t activate_type = 0;
    if (param.activation_param.has_active) {
        switch (param.activation_param.active_type) {
            case lite_api::ActivationType::kRelu:
            case lite_api::ActivationType::kRelu6:
            case lite_api::ActivationType::kLeakyRelu: {
                activate_type = (uint16_t)param.activation_param.active_type;
            } break;
            case lite_api::ActivationType::kHardSwish: {
                activate_type = (uint16_t)param.activation_param.active_type;
            } break;
            default: { LOG(FATAL) << "Conv2d: cannot support the activate type"; } break;
        }
    }
    // relu
    ActivationMetalParam activation_params{(unsigned short)activate_type, 0.0, 0.0, 0.0, 0.0, 0.0};
    switch (param.activation_param.active_type) {
        case lite_api::ActivationType::kIndentity:
        case lite_api::ActivationType::kRelu:
            break;
        case lite_api::ActivationType::kRelu6: {
            activation_params.threshold = param.activation_param.threshold;
        } break;
        case lite_api::ActivationType::kLeakyRelu: {
            activation_params.alpha = param.activation_param.Leaky_relu_alpha;
        } break;
        case lite_api::ActivationType::kHardSwish: {
            activation_params.threshold = param.activation_param.hard_swish_threshold;
            activation_params.offset = param.activation_param.hard_swish_offset;
            activation_params.scale = param.activation_param.hard_swish_scale;
        } break;
        default:
            break;
    }

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
        element_params,
        activation_params};

    params_buffer_ = std::make_shared<MetalBuffer>(metal_context_, sizeof(metalParam), &metalParam);

    // attention!!! filter: CNHW2NCHW
    auto rawdata = param.filter->data<float>();
    auto dims = filter_metal_dims_;
    auto tensorDim = DDimLite({dims[1], dims[0], dims[2], dims[3]});
    auto count = tensorDim.production();

    void* convertedPointer = TargetWrapperMetal::Malloc(count * sizeof(float));
    TargetWrapperMetal::MemsetSync(convertedPointer, 0, count * sizeof(float));
    auto weightsPointer = (float*)rawdata;
    auto transposed = (float*)convertedPointer;

    int index = 0;
    int order[4] = {1, 0, 2, 3};
    int index_order[4] = {0, 0, 0, 0};

    for (int d = 0; d < tensorDim[order[0]]; d++) {
        index_order[order[0]] = d;
        for (int c = 0; c < tensorDim[order[1]]; c++) {
            index_order[order[1]] = c;
            for (int b = 0; b < tensorDim[order[2]]; b++) {
                index_order[order[2]] = b;
                for (int a = 0; a < tensorDim[order[3]]; a++) {
                    index_order[order[3]] = a;
                    int tIndex = int(
                        index_order[3] +
                        tensorDim[3] *
                            (index_order[2] +
                                tensorDim[2] * (index_order[1] + tensorDim[1] * (index_order[0]))));
                    transposed[index] = weightsPointer[tIndex];
                    index += 1;
                }
            }
        }
    }
    // is depthwise-conv?
    bool pad_when_one_ch =
        !(filter_metal_dims_[1] == 1 && filter_metal_dims_[0] == param.x->dims()[1]);
    filter_buffer_ = std::make_shared<MetalBuffer>(metal_context_, filter_metal_dims_);
    filter_buffer_->pad_when_one_channel_ = pad_when_one_ch;
    filter_buffer_->CopyFromNCHW<float>(transposed);
    TargetWrapperMetal::Free(convertedPointer);

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];

    // 4x4
    if (function_name_ == "conv_transpose4x4_caculate") {
        // left
        auto left_dims = DDimLite({param.output->dims()[0],
            param.output->dims()[1],
            4 * param.x->dims()[2],
            4 * param.x->dims()[3]});
        intermediate_shift_left_ = new MetalImage(metal_context_, left_dims);
        intermediate_shift_left_->initImage(metal_context_);
        pipline_shift_left_ = [backend pipline:"conv_transpose4x4_stride2_shift_left"];

        // right
        auto right_dims = DDimLite({param.output->dims()[0],
            param.output->dims()[1],
            4 * param.x->dims()[2],
            2 * param.x->dims()[3]});
        intermediate_shift_right_ = new MetalImage(metal_context_, right_dims);
        intermediate_shift_right_->initImage(metal_context_);
        pipline_shift_right_ = [backend pipline:"conv_transpose4x4_stride2_shift_top"];

        // bias&relu
        auto output_dims = param.output->dims();
        intermediate_bias_relu_output_ = new MetalImage(metal_context_, output_dims);
        intermediate_bias_relu_output_->initImage(metal_context_);
        pipline_bias_relu_output_ = [backend pipline:"conv_transpose4x4_stride2_bias_relu"];
    }
}

void Conv2dTransposeImageCompute::run_without_mps() {
    if (function_name_ == "conv_transpose2x2_stride2") {
        run_2x2();
    } else if (HasSuffix(function_name_, "conv_transpose3x3_stride2x2")) {
        run_3x3();
    } else if (function_name_ == "conv_transpose4x4_caculate") {
        run_4x4();
    } else {
        LOG(ERROR) << "conv_transpose still cannot support this";
    }
}

void Conv2dTransposeImageCompute::run_2x2() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];
    [encoder setBuffer:(filter_buffer_->buffer()) offset:(0) atIndex:(1)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void Conv2dTransposeImageCompute::run_3x3() {
    const auto& param = this->Param<param_t>();
    auto pipline = pipline_;
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

    [backend dispatchEncoder:encoder
                     pipline:pipline
                threadsShape:@[
                    @(param.output->dims()[1]),
                    @(param.x->dims()[2]),
                    @(param.x->dims()[3])
                ]];
    [backend commit];
}

void Conv2dTransposeImageCompute::run_4x4() {
    const auto& param = this->Param<param_t>();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    do {
        auto pipline = pipline_;
        auto encoder = [backend commandEncoder];
        [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
        [encoder setTexture:(intermediate_shift_left_->image()) atIndex:(1)];
        [encoder setBuffer:(filter_buffer_->buffer()) offset:(0) atIndex:(0)];

        [backend dispatchEncoder:encoder
                         pipline:pipline
                    threadsShape:@[
                        @(param.output->dims()[1]),
                        @(param.x->dims()[2]),
                        @(param.x->dims()[3])
                    ]];
        [backend commit];
    } while (0);

    do {
        auto pipline = pipline_shift_left_;
        auto encoder = [backend commandEncoder];
        [encoder setTexture:(intermediate_shift_left_->image()) atIndex:(0)];
        [encoder setTexture:(intermediate_shift_right_->image()) atIndex:(1)];

        [backend dispatchEncoder:encoder
                         pipline:pipline
                    threadsShape:@[
                        @(param.output->dims()[1]),
                        @(param.x->dims()[2]),
                        @(param.x->dims()[3])
                    ]];
        [backend commit];
    } while (0);


    do {
        auto pipline = pipline_shift_right_;
        auto encoder = [backend commandEncoder];
        [encoder setTexture:(intermediate_shift_right_->image()) atIndex:(0)];
        [encoder setTexture:(intermediate_bias_relu_output_->image()) atIndex:(1)];

        [backend dispatchEncoder:encoder
                         pipline:pipline
                    threadsShape:@[
                        @(param.output->dims()[1]),
                        @(param.x->dims()[2]),
                        @(param.x->dims()[3])
                    ]];
        [backend commit];
    } while (0);

    do {
        auto pipline = pipline_bias_relu_output_;
        auto outTexture = output_buffer_->image();

        auto encoder = [backend commandEncoder];
        [encoder setTexture:(intermediate_bias_relu_output_->image()) atIndex:(0)];
        if (param.bias) {
            [encoder setTexture:(bias_buffer_->image()) atIndex:(1)];
        } else {
            [encoder setTexture:(blank_buffer_->image()) atIndex:(1)];
        }
        [encoder setTexture:(output_buffer_->image()) atIndex:(2)];
        [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

        [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
        [backend commit];
    } while (0);
}

#pragma mark - MPS

void Conv2dTransposeImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_conv_trans_op_) {
        if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
            [((__bridge MPSCNNConvolutionTranspose*)mps_conv_trans_op_)
                encodeToCommandBuffer:cmdbuf
                          sourceImage:(__bridge MPSImage*)mps_input_image_
                     destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
}

void Conv2dTransposeImageCompute::setup_with_mps() {
    const auto& param = this->Param<param_t>();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto padding_top = (*param.paddings)[0];
    auto padding_left = (*param.paddings)[2];

    int offsetX =
        static_cast<int>(param.filter->dims()[3] / 2 - param.filter->dims()[3] + 1 + padding_left);
    int offsetY =
        static_cast<int>(param.filter->dims()[2] / 2 - param.filter->dims()[2] + 1 + padding_top);

    auto rawdata = param.filter->data<float>();
    auto dims = filter_metal_dims_;                                   //
    auto tensorDim = DDimLite({dims[0], dims[1], dims[2], dims[3]});  //
    auto count = tensorDim.production();

    void* convertedPointer = TargetWrapperMetal::Malloc(count * sizeof(float));
    TargetWrapperMetal::MemsetSync(convertedPointer, 0, count * sizeof(float));
    auto weightsPointer = (float*)rawdata;
    auto transposed = (float*)convertedPointer;

    int length_nhw = dims[0] * dims[2] * dims[3];
    int length_chw = dims[1] * dims[2] * dims[3];
    int length_hw = dims[2] * dims[3];

    for (int n = 0; n < dims[0]; n++) {
        for (int c = 0; c < dims[1]; c++) {
            for (int h = 0; h < dims[2]; h++) {
                for (int w = 0; w < dims[3]; w++) {
                    int tIndex = h * dims[3] + w + length_nhw * c + length_hw * n;
                    int index = length_chw * n + (dims[2] - 1 - h) * dims[3] * dims[1] +
                                (dims[3] - 1 - w) * dims[1] + c;
                    transposed[index] = weightsPointer[tIndex];
                }
            }
        }
    }
    // mps-Convolution
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        output_buffer_->use_mps_ = true;
        const_cast<MetalImage*>(input_buffer_)->use_mps_ = true;
        auto filter_h = static_cast<int>(param.filter->dims()[2]);
        auto filter_w = static_cast<int>(param.filter->dims()[3]);
        auto input_c = MAX(4, static_cast<int>(input_buffer_->tensor_dim_[1]));
        auto output_c = MAX(4, static_cast<int>(output_buffer_->tensor_dim_[1]));

        MPSCNNConvolutionDescriptor* description =
            [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:filter_w
                                                                    kernelHeight:filter_h
                                                            inputFeatureChannels:input_c
                                                           outputFeatureChannels:output_c];

        description.strideInPixelsX = param.strides[0];
        description.strideInPixelsY = param.strides[1];
        description.dilationRateX = (*param.dilations)[0];
        description.dilationRateY = (*param.dilations)[1];
        description.groups = 1;

        MPSConvDataSource* scoure = [[MPSConvDataSource alloc] init];
        scoure.descriptor = description;
        filter_buffer_ = std::make_shared<MetalBuffer>(
            metal_context_, filter_metal_dims_, METAL_PRECISION_TYPE::HALF);
        filter_buffer_->convert_to_nhwc_ = false;
        filter_buffer_->CopyFromNCHW<float>(transposed);
        scoure.weights = filter_buffer_->rawdata();
        if (param.bias && canMPSAddByChannel()) {
            if (bias_buffer_->src_tensor_) {
                lite::Tensor* y = (lite::Tensor*)(bias_buffer_->src_tensor_);
                auto bias = y->data<float>();
                scoure.biasTerms = const_cast<float*>(bias);
            }
        }
        mps_conv_trans_op_ = (__bridge_retained void*)[[MPSCNNConvolutionTranspose alloc]
            initWithDevice:backend.device
                   weights:scoure];
        ((__bridge MPSCNNConvolutionTranspose*)mps_conv_trans_op_).offset =
            MPSOffset{.x = 0, .y = 0, .z = 0};
        ((__bridge MPSCNNConvolutionTranspose*)mps_conv_trans_op_).edgeMode = MPSImageEdgeModeZero;
        ((__bridge MPSCNNConvolutionTranspose*)mps_conv_trans_op_).kernelOffsetX = offsetX;
        ((__bridge MPSCNNConvolutionTranspose*)mps_conv_trans_op_).kernelOffsetY = offsetY;

        // MPS input and output
        mps_input_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_->image()
                                                       featureChannels:input_c];
        mps_output_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                       featureChannels:output_c];
    }
}

#pragma mark - internal

std::string Conv2dTransposeImageCompute::KernelFunctionName(const param_t& param) {
    if (filter_metal_dims_[3] == 2 && filter_metal_dims_[2] == 2) {
        if (param.strides[0] == 2 && param.strides[1] == 2) {
            return "conv_transpose2x2_stride2";
        }
    } else if (filter_metal_dims_[3] == 3 && filter_metal_dims_[2] == 3) {
        if (param.strides[0] == 2 && param.strides[1] == 2) {
            if (filter_metal_dims_[0] == 1 && filter_metal_dims_[1] == param.x->dims()[1] &&
                param.groups == param.x->dims()[1]) {
                return "depthwise_conv_transpose3x3_stride2x2";
            } else if (filter_metal_dims_[1] == param.x->dims()[1]) {
                return "conv_transpose3x3_stride2x2";
            }
        }
    } else if (filter_metal_dims_[3] == 4 && filter_metal_dims_[2] == 4) {
        if (param.strides[0] == 2 && param.strides[1] == 2) {
            return "conv_transpose4x4_caculate";
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

bool Conv2dTransposeImageCompute::HasSuffix(const std::string& function_name,
    const std::string& suffix) {
    auto s_size = suffix.size();
    auto f_size = function_name.size();
    if (f_size >= s_size && function_name.compare(f_size - s_size, s_size, suffix) == 0) {
        return true;
    }
    return false;
}

bool Conv2dTransposeImageCompute::canAddUseMPS() {
    return canMPSAddByChannel() || canMPSAddByElement();
}

bool Conv2dTransposeImageCompute::canMPSAddByChannel() {
    if (!bias_buffer_->src_tensor_) {
        return false;
    }
    lite::Tensor* y = (lite::Tensor*)(bias_buffer_->src_tensor_);
    if (y->dims().size() == 1) {
        return true;
    }
    return false;
}

bool Conv2dTransposeImageCompute::canMPSAddByElement() {
    const auto& param = this->Param<param_t>();
    lite::Tensor* y = (lite::Tensor*)(bias_buffer_->src_tensor_);
    if (y->dims() == param.output->dims()) {
        return true;
    }
    return false;
}

void Conv2dTransposeImageCompute::release_memory() {
    TargetWrapperMetal::FreeImage(output_buffer_);
    TargetWrapperMetal::FreeImage(blank_buffer_);
}

void Conv2dTransposeImageCompute::release_intermediate() {
    TargetWrapperMetal::FreeImage(intermediate_shift_left_);
    TargetWrapperMetal::FreeImage(intermediate_shift_right_);
    TargetWrapperMetal::FreeImage(intermediate_bias_relu_output_);
}

void Conv2dTransposeImageCompute::release_mps_memory() {
}

Conv2dTransposeImageCompute::~Conv2dTransposeImageCompute() {
    release_memory();
    release_mps_memory();
    release_intermediate();
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
