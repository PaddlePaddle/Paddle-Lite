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

#include "lite/kernels/metal/image_op/conv2d_transpose_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void Conv2dTransposeImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.output->dims();
    auto input_dims = param.x->dims();
    input_buffer_ = param.x->template data<P, MetalImage>();
    if (param.bias) bias_buffer_ = param.bias->template data<P, MetalImage>();

    if (param.activation_param.has_active) {
        if (lite_api::ActivationType::kRelu == param.activation_param.active_type)
            activate_type_ = 1;
        else if (lite_api::ActivationType::kRelu6 == param.activation_param.active_type) {
            activate_type_ = 2;
            relu6_thredhold_ = static_cast<short>(param.activation_param.hard_swish_threshold);
        } else {
            throw std::logic_error("cannot support the activate type");
        }
    }

    output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);

    auto* blank_host = (float*)malloc(sizeof(float) * output_dims[1]);
    memset(blank_host, 0, sizeof(float) * output_dims[1]);
    DDim blank_dim = DDimLite({output_dims[1]});
    blank_tensor_.Resize(blank_dim);
    auto p = blank_tensor_.mutable_data<P, MetalImage>(blank_dim);
    p->template CopyFromNCHW<float>(blank_host);
    free(blank_host);

    function_name_ = KernelFunctionName(param, metal_context_->use_aggressive_optimization());
    if (function_name_.empty()) {
        throw std::logic_error("ERROR: cannot find the kernel name of this conv2d_transpose");
    }

    if (activate_type_ == 2) {
        auto index = function_name_.find("relu");
        if (index != -1) function_name_.replace(index, 4, "relu6");
    }

    SetupWithoutMPS();
    kernel_ = metal_context_->GetKernel(*device, function_name_);
    queue_ = metal_context_->GetDefaultQueue(*device);
}

template <typename P, PrecisionType PTYPE>
void Conv2dTransposeImageCompute<P, PTYPE>::Run() {
    const auto& param = this->template Param<param_t>();
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0)atIndex:(0)];
    [encoder->metal_command_encoder_ setBuffer:(filter_buffer_->buffer()) offset:(0)atIndex:(1)];

    kernel_->Execute(*encoder, global_work_size, false);
}

template <typename P, PrecisionType PTYPE>
std::string Conv2dTransposeImageCompute<P, PTYPE>::KernelFunctionName(
    const param_t& param, bool use_aggressive_optimization) {
    if (std::is_same<float, P>::value) {
        if (param.filter->dims()[3] == 2 && param.filter->dims()[2] == 2) {
            if (param.strides[0] == 2 && param.strides[1] == 2) {
                return "conv_transpose2x2_stride2";
            }
        } else if (param.filter->dims()[3] == 3 && param.filter->dims()[2] == 3) {
            if (param.strides[0] == 2 && param.strides[1] == 2) {
                return "conv_transpose3x3_caculate";
            }
        }
        return "";
    } else if (std::is_same<MetalHalf, P>::value) {
        if (param.filter->dims()[3] == 2 && param.filter->dims()[2] == 2) {
            if (param.strides[0] == 2 && param.strides[1] == 2) {
                return "conv_transpose2x2_stride2_half";
            }
        } else if (param.filter->dims()[3] == 3 && param.filter->dims()[2] == 3) {
            if (param.strides[0] == 2 && param.strides[1] == 2) {
                return "conv_transpose3x3_stride2x2_half";
            }
        }
        return "";
    }
}

template <typename P, PrecisionType PTYPE>
bool Conv2dTransposeImageCompute<P, PTYPE>::HasPrefix(const std::string& function_name,
                                                      const std::string& prefix) {
    if (function_name.size() >= prefix.size() &&
        function_name.compare(0, prefix.size(), prefix) == 0) {
        return true;
    }
    return false;
}

template <typename P, PrecisionType PTYPE>
void Conv2dTransposeImageCompute<P, PTYPE>::SetupWithMPS() {
    // TODO: (lzy) add MPS support
}

template <typename P, PrecisionType PTYPE>
void Conv2dTransposeImageCompute<P, PTYPE>::SetupWithoutMPS() {
    const auto& param = this->template Param<param_t>();
    auto padLeft = (*param.paddings)[2];
    auto padTop = (*param.paddings)[0];
    assert((*param.paddings)[0] == (*param.paddings)[1]);

    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

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

    auto filter_buffer = param.filter->template data<float>();
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

    ElementwiseAddMetalParam addParam = {
        params_fast,
        add_by_channel,
        params_axis,
        (int)output_buffer_->tensor_dim_.size(),
        {xdim[0], xdim[1], xdim[2], xdim[3]},
        {output_buffer_->transpose_[0], output_buffer_->transpose_[1],
         output_buffer_->transpose_[2], output_buffer_->transpose_[3]},
        {ydim[0], ydim[1], ydim[2], ydim[3]},
        {bias_buffer_->transpose_[0], bias_buffer_->transpose_[1], bias_buffer_->transpose_[2],
         bias_buffer_->transpose_[3]}};
    ConvTransposeAddMetalParam metalParam = {kernelWidth, kernelHeight, strideX,   strideY,
                                             paddingX,    paddingY,     dilationX, dilationY,
                                             groups,      inputC,       filterC,   outputC,
                                             hasAdd,      addParam};

    params_buffer_ = metal_context_->CreateBuffer(*device, &metalParam, sizeof(metalParam),
                                                  METAL_ACCESS_FLAG::CPUWriteOnly);

    if (HasPrefix(function_name_, "conv_transpose2x2")) {
        filter_buffer_ =
            std::make_shared<MetalBuffer>(*device, param.filter->dims(), METAL_PRECISION_TYPE::HALF,
                                          false, false, true);
    } else {
        throw std::logic_error("ERROR: conv_transpose still cannot support this");
    }
    filter_buffer_->CopyFromNCHW<float>(filter_buffer);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::Conv2dTransposeImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::Conv2dTransposeImageCompute<MetalHalf,
                                                                         PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::Conv2dTransposeImageCompute<float, PRECISION(kFloat)>
    MetalConv2dTransposeFp32;
typedef paddle::lite::kernels::metal::Conv2dTransposeImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalConv2dTransposeFp16;

// TODO:(lzy) need debug to open the kernel
// REGISTER_LITE_KERNEL(conv2d_transpose,
//                     kMetal,
//                     kFloat,
//                     kMetalTexture2DArray,
//                     MetalConv2dTransposeFp32,
//                     def)
//    .BindInput("Input",
//               {LiteType::GetTensorTy(TARGET(kMetal),
//                                      PRECISION(kFloat),
//                                      DATALAYOUT(kMetalTexture2DArray))})
//    .BindInput("Bias",
//               {LiteType::GetTensorTy(TARGET(kMetal),
//                                      PRECISION(kFloat),
//                                      DATALAYOUT(kMetalTexture2DArray))})
//    .BindInput("Filter",
//               {LiteType::GetTensorTy(TARGET(kHost),
//                                      PRECISION(kFloat),
//                                      DATALAYOUT(kNCHW))})
//    .BindOutput("Output",
//                {LiteType::GetTensorTy(TARGET(kMetal),
//                                       PRECISION(kFloat),
//                                       DATALAYOUT(kMetalTexture2DArray))})
//    .Finalize();
//
// REGISTER_LITE_KERNEL(conv2d_transpose,
//                     kMetal,
//                     kFP16,
//                     kMetalTexture2DArray,
//                     MetalConv2dTransposeFp16,
//                     def)
//    .BindInput("Input",
//               {LiteType::GetTensorTy(TARGET(kMetal),
//                                      PRECISION(kFP16),
//                                      DATALAYOUT(kMetalTexture2DArray))})
//    .BindInput("Bias",
//               {LiteType::GetTensorTy(TARGET(kMetal),
//                                      PRECISION(kFP16),
//                                      DATALAYOUT(kMetalTexture2DArray))})
//    .BindInput("Filter",
//               {LiteType::GetTensorTy(TARGET(kHost),
//                                      PRECISION(kFloat),
//                                      DATALAYOUT(kNCHW))})
//    .BindOutput("Output",
//                {LiteType::GetTensorTy(TARGET(kMetal),
//                                       PRECISION(kFP16),
//                                       DATALAYOUT(kMetalTexture2DArray))})
//    .Finalize();
