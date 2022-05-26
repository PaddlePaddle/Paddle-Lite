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

#include "lite/kernels/metal/image_op/conv2d_image_compute.h"
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

void Conv2dImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    init_memory();
    init_for_run();
}

void Conv2dImageCompute::ReInitWhenNeeded() {
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

void Conv2dImageCompute::init_memory() {
    const auto& param = this->Param<param_t>();
    auto input_dims = param.x->dims();
    auto filter_dims = param.filter->dims();
    auto output_dims = param.output->dims();

    int filter_n = static_cast<int>(filter_dims[0]);
    int filter_c = static_cast<int>(filter_dims[1]);
    int input_c = static_cast<int>(input_dims[1]);
    is_depthwise_ = filter_c == 1 && filter_n == input_c;

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

void Conv2dImageCompute::init_for_run() {
    const auto& param = this->Param<param_t>();
    function_name_ =
        KernelFunctionName(param, metal_context_->use_winograde(), metal_context_->use_quadruple());
    // use mps or not
    bool should_use_mps = false;
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        if (metal_context_->use_mps()) {
            int input_c = static_cast<int>(input_buffer_->tensor_dim_[1]);
            int output_c = static_cast<int>(output_buffer_->tensor_dim_[1]);
            // input channel must >=3
            // attention: should be >=4, texture data layout is RGBA
            if (is_depthwise_) {
                if (input_c >= 3 && output_c >= 3) {
                    should_use_mps = true;
                }
            } else {
                if (input_c >= 3) {
                    should_use_mps = true;
                }
            }
        }
    }
    if (IsWinoGrad(function_name_) || IsQuadruple(function_name_)) {
        should_use_mps = false;
    }
    if (param.bias) {
        if (!canMPSAddByChannel()) {
            should_use_mps = false;
        }
    }

    // MPS don't support LeakyRelu and HardSwish
    switch (param.activation_param.active_type) {
        case lite_api::ActivationType::kIndentity:
        case lite_api::ActivationType::kRelu:
            break;
        case lite_api::ActivationType::kRelu6:
            break;
        case lite_api::ActivationType::kHardSigmoid:
            break;
        case lite_api::ActivationType::kPRelu:
            break;
        case lite_api::ActivationType::kHardSwish:
        case lite_api::ActivationType::kLeakyRelu:
            should_use_mps = NO;
            break;
        default:
            break;
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

void Conv2dImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF

void Conv2dImageCompute::run_without_mps() {
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

    bool quadruple = false;
    if (IsWinoGrad(function_name_) || IsQuadruple(function_name_)) {
        quadruple = true;
    }

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture quadruple:quadruple];
    [backend commit];
}

std::string Conv2dImageCompute::KernelFunctionName(const param_t& param,
    bool use_winograde,
    bool use_quadruple) {
    auto filter_n = param.filter->dims()[0];
    auto filter_c = param.filter->dims()[1];
    auto filter_h = param.filter->dims()[2];
    auto filter_w = param.filter->dims()[3];
    auto input_c = param.x->dims()[1];

    // depthwise_conv
    if (filter_c == 1 && filter_n == input_c) {
        if (filter_w == 3 && filter_h == 3) {
#ifdef LITE_WITH_METAL_FULL
#else
            if (use_winograde) {
                bool winograd = (filter_w == 3) && (filter_h == 3) && (param.strides[0] == 1) &&
                                (param.strides[1] == 1) && ((*param.dilations)[0] == 1) &&
                                ((*param.dilations)[1] == 1) && ((*param.paddings)[2] == 1) &&
                                ((*param.paddings)[0] == 1);  // paddings: top bottom left right
                if (winograd) {
                    return "depthwise_conv_3x3_winograd";
                }
            }
#endif
            return "depthwise_conv_3x3";
        } else if (filter_w == 5 && filter_h == 5) {
            return "depthwise_conv_5x5";
        }
        return "";
    }
    // conv
    else {
        if (filter_w == 1 && filter_h == 1) {
            if (use_quadruple) {
                auto padTop = (*param.paddings)[0];
                auto padLeft = (*param.paddings)[2];
                if (filter_c <= 16 && padTop == 0 && padLeft == 0) {
                    return "conv_1x1_quadruple";
                }
            }
            return "conv_1x1";
        } else if (filter_w == 3 && filter_h == 3) {
            if (param.groups == 1) {
#ifdef LITE_WITH_METAL_FULL
#else
                if (use_winograde) {
                    bool winograd = (filter_w == 3) && (filter_h == 3) && (param.strides[0] == 1) &&
                                    (param.strides[1] == 1) && ((*param.dilations)[0] == 1) &&
                                    ((*param.dilations)[1] == 1) && ((*param.paddings)[2] == 1) &&
                                    ((*param.paddings)[0] == 1);  // paddings: top bottom left right
                    if (winograd) {
                        return "conv_3x3_winograd";
                    }
                }
#endif
                return "conv_3x3";
            } else if ((input_c == (filter_c * param.groups)) && filter_n == input_c) {
                return "group_conv_3x3";
            } else {
                return "depthwise_conv_3x3_unequal";
            }
        } else if (filter_w == 1 && filter_h == 5) {
            return "conv_5x1";
        } else if (filter_w == 5 && filter_h == 1) {
            return "conv_1x5";
        } else if (filter_w == 7 && filter_h == 7) {
            return "conv_7x7";
        } else {
            return "";
        }
    }
    return "";
}

void Conv2dImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    auto padTop = (*param.paddings)[0];
    auto padLeft = (*param.paddings)[2];
    assert((*param.paddings)[0] == (*param.paddings)[1]);

    int offsetX = static_cast<int>(
        ((int)((*param.dilations)[1]) * (param.filter->dims()[3] - 1) + 1) / 2 - padLeft);
    int offsetY = static_cast<int>(
        ((int)((*param.dilations)[0]) * (param.filter->dims()[2] - 1) + 1) / 2 - padTop);
    float offsetZ = 0.0;

    int iC = static_cast<int>(param.x->dims()[1]);
    int fC = static_cast<int>(param.filter->dims()[1]);
    int oC = static_cast<int>(param.output->dims()[1]);

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
    // set shader params
    MetalConvParam conv_params{(short)offsetX,
        (short)offsetY,
        (short)offsetZ,
        (unsigned short)(param.strides[1]),
        (unsigned short)(param.strides[0]),
        (unsigned short)((*param.dilations)[1]),
        (unsigned short)((*param.dilations)[0]),
        (unsigned short)(param.groups),
        (unsigned short)(iC),
        (unsigned short)(fC),
        (unsigned short)(oC),
        (unsigned short)(param.bias ? 1 : 0),
        (unsigned short)(param.activation_param.has_active ? 1 : 0),
        element_params,
        activation_params};

    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(conv_params), &conv_params);

    auto filter = param.filter->data<float>();
    if (IsWinoGrad(function_name_)) {
        // data convert
        DataConverter<float>* converter = new WinogradPointerConverter<float>();
        auto from_dim = param.filter->dims();
        auto to_capacity = converter->Capacity(from_dim);
        auto to_filter = (float*)TargetWrapperMetal::Malloc(sizeof(float) * to_capacity);
        try {
            converter->Convert(const_cast<float*>(filter), to_filter, from_dim);
        } catch (std::exception& error) {
            TargetWrapperMetal::Free(to_filter);
            LOG(FATAL) << "metal_conv2d: still not finish winograd";
        }
        auto to_dim = converter->GetToDim(from_dim);
        filter_buffer_ =
            std::make_shared<MetalBuffer>(metal_context_, to_dim, METAL_PRECISION_TYPE::HALF);
        if (function_name_ == "conv_3x3_winograd") {
            filter_buffer_->convert_to_nhwc_ = false;
            filter_buffer_->pad_when_one_channel_ = false;
        } else {
            filter_buffer_->convert_to_nhwc_ = true;
            bool pad_when_one_ch =
                !(param.filter->dims()[1] == 1 && param.filter->dims()[0] == param.x->dims()[1]);
            filter_buffer_->pad_when_one_channel_ = pad_when_one_ch;
        }
        filter_buffer_->CopyFromNCHW<float>(to_filter);
        TargetWrapperMetal::Free(to_filter);
    } else {
        bool pad_when_one_ch =
            !(param.filter->dims()[1] == 1 && param.filter->dims()[0] == param.x->dims()[1]);
        filter_buffer_ = std::make_shared<MetalBuffer>(metal_context_, param.filter->dims());
        if (param.groups != 1 && param.filter->dims()[0] != param.x->dims()[1]) {
            filter_buffer_->pad_when_one_channel_ = false;
        } else
            filter_buffer_->pad_when_one_channel_ = pad_when_one_ch;
        filter_buffer_->CopyFromNCHW<float>(filter);
    }

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

void Conv2dImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_conv_op_) {
        if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
            [((__bridge MPSCNNConvolution*)mps_conv_op_)
                encodeToCommandBuffer:cmdbuf
                          sourceImage:(__bridge MPSImage*)mps_input_image_
                     destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
}

void Conv2dImageCompute::setup_with_mps() {
    const auto& param = this->Param<param_t>();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto padding_top = (*param.paddings)[0];
    auto padding_left = (*param.paddings)[2];
    int offsetX = static_cast<int>(
        ((int)((*param.dilations)[1]) * (param.filter->dims()[3] - 1) + 1) / 2 - padding_left);
    int offsetY = static_cast<int>(
        ((int)((*param.dilations)[0]) * (param.filter->dims()[2] - 1) + 1) / 2 - padding_top);

    // mps-Convolution
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        output_buffer_->use_mps_ = true;
        const_cast<MetalImage*>(input_buffer_)->use_mps_ = true;
        auto filter_h = static_cast<int>(param.filter->dims()[2]);
        auto filter_w = static_cast<int>(param.filter->dims()[3]);
        auto input_c = static_cast<int>(input_buffer_->tensor_dim_[1]);
        auto output_c = fmax(4, static_cast<int>(output_buffer_->tensor_dim_[1]));
        MPSCNNConvolutionDescriptor* description = nil;
        if (is_depthwise_) {
            description = [MPSCNNDepthWiseConvolutionDescriptor
                cnnConvolutionDescriptorWithKernelWidth:filter_w
                                           kernelHeight:filter_h
                                   inputFeatureChannels:input_c
                                  outputFeatureChannels:output_c];
        } else {
            description =
                [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:filter_w
                                                                        kernelHeight:filter_h
                                                                inputFeatureChannels:input_c
                                                               outputFeatureChannels:output_c];
        }
        description.strideInPixelsX = param.strides[0];
        description.strideInPixelsY = param.strides[1];
        description.dilationRateX = (*param.dilations)[0];
        description.dilationRateY = (*param.dilations)[1];
        if (!is_depthwise_) description.groups = param.groups;
        // active function
        switch (param.activation_param.active_type) {
            case lite_api::ActivationType::kRelu: {
                description.fusedNeuronDescriptor =
                    [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU a:0.0];
            } break;
            case lite_api::ActivationType::kRelu6: {
                description.fusedNeuronDescriptor = [MPSNNNeuronDescriptor
                    cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLUN
                                              a:0.0
                                              b:param.activation_param.threshold];
            } break;
            case lite_api::ActivationType::kHardSigmoid: {
                description.fusedNeuronDescriptor = [MPSNNNeuronDescriptor
                    cnnNeuronDescriptorWithType:MPSCNNNeuronTypeHardSigmoid
                                              a:param.activation_param.hard_sigmoid_slope
                                              b:param.activation_param.hard_sigmoid_offset];
            } break;
            case lite_api::ActivationType::kPRelu: {
                description.fusedNeuronDescriptor =
                    [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypePReLU a:0.0];
            } break;
            default:
                break;
        }
        // MPS op
        MPSConvDataSource* scoure = [[MPSConvDataSource alloc] init];
        scoure.descriptor = description;
        // mps weights(filter) NHWC
        // weight: [ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels / groups ]
        auto filter = param.filter->data<float>();
        DataConverter<float>* converter = new MPSPointerConverter<float>();
        auto from_dim = param.filter->dims();
        auto count = from_dim.production();
        auto to_filter = (float*)TargetWrapperMetal::Malloc(sizeof(float) * count);
        try {
            converter->Convert(const_cast<float*>(filter), to_filter, from_dim);
        } catch (std::exception& error) {
            TargetWrapperMetal::Free(to_filter);
            TargetWrapperMetal::Free(converter);
            LOG(FATAL) << "metal_conv2d: still not finish mps";
        }
        filter_buffer_ = std::make_shared<MetalBuffer>(
            metal_context_, param.filter->dims(), METAL_PRECISION_TYPE::HALF);
        filter_buffer_->convert_to_nhwc_ = false;
        filter_buffer_->CopyFromNCHW<float>(to_filter);
        TargetWrapperMetal::Free(to_filter);
        TargetWrapperMetal::Free(converter);
        scoure.weights = filter_buffer_->rawdata();
        // bias
        if (param.bias && canMPSAddByChannel()) {
            if (bias_buffer_->src_tensor_) {
                lite::Tensor* y = (lite::Tensor*)(bias_buffer_->src_tensor_);
                auto bias = y->data<float>();
                scoure.biasTerms = const_cast<float*>(bias);
            }
        }
        mps_conv_op_ =
            (__bridge_retained void*)[[MPSCNNConvolution alloc] initWithDevice:backend.device
                                                                       weights:scoure];
        ((__bridge MPSCNNConvolution*)mps_conv_op_).offset = MPSOffset{.x = offsetX, .y = offsetY};
        ((__bridge MPSCNNConvolution*)mps_conv_op_).edgeMode = MPSImageEdgeModeZero;
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

bool Conv2dImageCompute::IsWinoGrad(const std::string& function_name) {
    std::string suffix = "winograd";
    if (function_name.size() >= suffix.size() &&
        function_name.compare(function_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return true;
    }
    return false;
}

bool Conv2dImageCompute::IsQuadruple(const std::string& function_name) {
    std::string suffix = "quadruple";
    if (function_name.size() >= suffix.size() && function_name.find(suffix) != std::string::npos) {
        return true;
    }
    return false;
}

bool Conv2dImageCompute::canAddUseMPS() {
    return canMPSAddByChannel() || canMPSAddByElement();
}

bool Conv2dImageCompute::canMPSAddByChannel() {
    if (!bias_buffer_->src_tensor_) {
        return false;
    }
    lite::Tensor* y = (lite::Tensor*)(bias_buffer_->src_tensor_);
    if (y->dims().size() == 1) {
        return true;
    }
    return false;
}

bool Conv2dImageCompute::canMPSAddByElement() {
    const auto& param = this->Param<param_t>();
    lite::Tensor* y = (lite::Tensor*)(bias_buffer_->src_tensor_);
    if (y->dims() == param.output->dims()) {
        return true;
    }
    return false;
}

void Conv2dImageCompute::release_memory() {
    TargetWrapperMetal::FreeImage(output_buffer_);
    TargetWrapperMetal::FreeImage(blank_buffer_);
}

void Conv2dImageCompute::release_mps_memory() {
    if (mps_conv_op_) {
        CFRelease(mps_conv_op_);
        mps_conv_op_ = nullptr;
    }
    if (mps_input_image_) {
        CFRelease(mps_input_image_);
        mps_input_image_ = nullptr;
    }
    if (mps_output_image_) {
        CFRelease(mps_output_image_);
        mps_output_image_ = nullptr;
    }
}

Conv2dImageCompute::~Conv2dImageCompute() {
    release_memory();
    release_mps_memory();
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#pragma mark -

REGISTER_LITE_KERNEL(conv2d,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::Conv2dImageCompute,
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

REGISTER_LITE_KERNEL(conv2d,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::Conv2dImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Filter",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Output",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
