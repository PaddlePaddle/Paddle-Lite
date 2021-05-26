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

#include "lite/kernels/metal/image_op/depthwise_conv2d_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/backends/metal/metal_debug.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void DepthwiseConv2dImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto device = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();
  input_buffer_ = param.x->template data<P, MetalImage>();
  if (param.bias) bias_buffer_ = param.bias->template data<P, MetalImage>();
  output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);

  if (param.activation_param.has_active) {
    if (lite_api::ActivationType::kRelu == param.activation_param.active_type)
      activate_type_ = 1;
    else if (lite_api::ActivationType::kRelu6 == param.activation_param.active_type) {
      activate_type_ = 2;
      relu6_thredhold_ = param.activation_param.hard_swish_threshold;
    } else {
      throw std::logic_error("cannot support the activate type");
    }
  }

  float* blank_host = (float*)malloc(sizeof(float) * output_dims[1]);
  memset(blank_host, 0, sizeof(float) * output_dims[1]);

  DDim blank_dim = DDimLite({output_dims[1]});
  blank_tensor_.Resize(blank_dim);
  auto p = blank_tensor_.mutable_data<P, MetalImage>(blank_dim);
  p->template CopyFromNCHW<float>(blank_host);
  free(blank_host);

  bool should_use_mps = false;
  function_name_ = KernelFunctionName(param, metal_context_->use_aggressive_optimization());

#ifdef TARGET_IOS
  if (@available(iOS 11.0, *)) {
#endif
    if (metal_context_->use_mps() || metal_context_->use_aggressive_optimization()) {
      if (input_dims[1] >= 3 && output_buffer_->tensor_dim_[1] >= 3) {
        // should_use_mps = true; //TODO: (lzy) add MPS support
      }
    }
#ifdef TARGET_IOS
  }
#endif
  if (IsWinoGrad(function_name_)) {
    should_use_mps = false;
  }

  int filter_channel = param.filter->dims()[1];
  int filter_n = param.filter->dims()[0];
  bool isDepthWise = filter_channel == 1 && filter_n == input_buffer_->tensor_dim_[1];
  if (!isDepthWise && param.groups > 1) {
    should_use_mps = false;
  }

  if (function_name_ == "") {
    throw std::logic_error("ERROR: cannot find the name of the depthwise_conv2d");
  }

  if (activate_type_ == 2) {
    auto index = function_name_.find("relu");
    if (index != -1) function_name_.replace(index, 4, "relu6");
  }

  kernel_ = metal_context_->GetKernel(*device, function_name_.c_str());
  queue_ = metal_context_->GetDefaultQueue(*device);

  if (should_use_mps) {
    SetupWithMPS();
  } else {
    SetupWithoutMPS();
  }
}

template <typename P, PrecisionType PTYPE>
void DepthwiseConv2dImageCompute<P, PTYPE>::Run() {
  const auto& param = this->template Param<param_t>();
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  {
    auto encoder = std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    if (param.bias) {
      [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
      [encoder->metal_command_encoder_ setTexture:(bias_buffer_->image()) atIndex:(1)];
      [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(2)];
      [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];
      [encoder->metal_command_encoder_ setBuffer:(filter_buffer_->buffer()) offset:(0) atIndex:(1)];

      bool quadruple = false;
      if (IsWinoGrad(function_name_) ||
          function_name_ == "conv_add_relu_1x1_quadruple_half") {
        quadruple = true;
      }
      kernel_->Execute(*encoder, global_work_size, quadruple);
    } else {
      auto blank_buffer = blank_tensor_.data<P, MetalImage>();
      [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
      [encoder->metal_command_encoder_ setTexture:(blank_buffer->image()) atIndex:(1)];
      [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(2)];
      [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];
      [encoder->metal_command_encoder_ setBuffer:(filter_buffer_->buffer()) offset:(0) atIndex:(1)];

      bool quadruple = false;
      if (IsWinoGrad(function_name_) ||
          function_name_ == "conv_add_relu_1x1_quadruple_half") {
        quadruple = true;
      }
      kernel_->Execute(*encoder, global_work_size, quadruple);
    }
  }
}

template <typename P, PrecisionType PTYPE>
string DepthwiseConv2dImageCompute<P, PTYPE>::KernelFunctionName(const param_t& param,
                                                                 bool use_aggressive_optimization) {
  auto filter_width = param.filter->dims()[3];
  auto filter_height = param.filter->dims()[2];
  auto filter_channel = param.filter->dims()[1];
  auto filter_n = param.filter->dims()[0];
  auto padLeft = (*param.paddings)[2];
  auto padTop = (*param.paddings)[0];
  auto dilations = (*param.dilations);

  auto input_tensor_dim = param.x->dims();

  if (std::is_same<P, float>::value) {
    if (filter_width == 1 && filter_height == 1) {
      return "conv_add_relu_1x1";
    } else if (filter_width == 3 && filter_height == 3) {
      if (filter_channel == 1 && filter_n == param.x->dims()[1]) {
        return "depthwise_conv_add_relu_3x3";
      } else {
        if (param.groups == 1) {
          return "conv_add_relu_3x3";
        } else {
          return "group_conv_add_relu_3x3";
        }
      }
    } else if (filter_width == 5 && filter_height == 5) {
      if (filter_channel == 1 && filter_n == param.x->dims()[1]) {
        return "depthwise_conv_add_relu_5x5";
      } else {
        if (param.groups == 1) {
          return "conv_add_relu_5x5";
        } else {
          return "group_conv_add_relu_5x5";
        }
      }
    } else if (filter_width == 1 && filter_height == 5) {
      return "conv_add_relu_5x1";
    } else if (filter_width == 5 && filter_height == 1) {
      return "conv_add_relu_1x5";
    } else if (filter_width == 7 && filter_height == 7) {
      return "conv_add_relu_7x7";
    } else {
      return "";
    }
  } else if (std::is_same<P, MetalHalf>::value) {
    if (filter_width == 1 && filter_height == 1) {
      if (filter_channel <= 16 && padLeft == 0 && padTop == 0) {
        return "conv_add_relu_1x1_quadruple_half";
      } else {
        return "conv_add_relu_1x1_half";
      }
    } else if (filter_width == 3 && filter_height == 3) {
      if (filter_channel == 1 && param.filter->dims()[0] == param.x->dims()[1]) {
        if (use_aggressive_optimization) {
          bool could_use_winograd = filter_width == 3 && filter_height == 3 &&
                                    param.strides[0] == 1 && param.strides[1] == 1 &&
                                    dilations[0] == 1 && dilations[1] == 1 && padLeft == 1 &&
                                    padTop == 1;
          if (could_use_winograd) {
            return "depthwise_conv_add_relu_3x3_half_winograd";
          }
        }
        return "depthwise_conv_add_relu_3x3_half";
      } else {
        if (param.groups == 1) {
          if (use_aggressive_optimization) {
            bool could_use_winograd = filter_width == 3 && filter_height == 3 &&
                                      param.strides[0] == 1 && param.strides[1] == 1 &&
                                      dilations[0] == 1 && dilations[1] == 1 && padLeft == 1 &&
                                      padTop == 1;
            if (could_use_winograd) {
              return "conv_add_relu_3x3_half_winograd";
            }
          }
          return "conv_add_relu_3x3_half";
        } else {
          return "group_conv_add_relu_3x3_half";
        }
      }
    } else if (filter_width == 5 && filter_height == 5) {
      if (filter_channel == 1 && filter_n == param.x->dims()[1]) {
        return "depthwise_conv_add_relu_5x5_half";
      } else {
        if (param.groups == 1) {
          return "conv_add_relu_5x5_half";
        } else {
          return "group_conv_add_relu_5x5_half";
        }
      }
    } else if (filter_width == 1 && filter_height == 5) {
      return "conv_add_relu_5x1_half";
    } else if (filter_width == 5 && filter_height == 1) {
      return "conv_add_relu_1x5_half";
    } else if (filter_width == 7 && filter_height == 7) {
      return "conv_add_relu_7x7_half";
    } else {
      return "";
    }
  }
}

template <typename P, PrecisionType PTYPE>
bool DepthwiseConv2dImageCompute<P, PTYPE>::IsWinoGrad(string function_name) {
  std::string suffix = "winograd";
  if (function_name.size() >= suffix.size() &&
      function_name.compare(
          function_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return true;
  }
  return false;
}

template <typename P, PrecisionType PTYPE>
void DepthwiseConv2dImageCompute<P, PTYPE>::SetupWithMPS() {
  // TODO: (lzy)
}

template <typename P, PrecisionType PTYPE>
void DepthwiseConv2dImageCompute<P, PTYPE>::SetupWithoutMPS() {
  const auto& param = this->template Param<param_t>();
  auto padLeft = (*param.paddings)[2];
  auto padTop = (*param.paddings)[0];
  assert((*param.paddings)[0] == (*param.paddings)[1]);

  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto device = metal_context_->GetDefaultDevice();

  int offsetX =
      ((int)((*param.dilations)[1]) * (param.filter->dims()[3] - 1) + 1) / 2 -
      (int)(padLeft);
  int offsetY =
      ((int)((*param.dilations)[0]) * (param.filter->dims()[2] - 1) + 1) / 2 -
      (int)(padTop);

  float offsetZ = 0.0;
  int iC = param.x->dims()[1];
  int fC = param.filter->dims()[1];
  int oC = param.output->dims()[1];

  if (param.bias) {
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
        (axis == 1 || (axis == -1 &&
                       bias_buffer_->tensor_dim_[0] ==
                           output_buffer_->pad_to_four_dim_[1]))) {
      add_by_channel = 1;
    }

    ElementwiseAddMetalParam element_params = {
        params_fast,
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

    MetalConvParam conv_params{
        (short)offsetX,
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
        element_params};

    params_buffer_ = metal_context_->CreateBuffer(*device,
                                           &conv_params,
                                           sizeof(conv_params),
                                           METAL_ACCESS_FLAG::CPUWriteOnly);
  } else {
    MetalConvParam conv_params{
        (short)offsetX,
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
        (unsigned short)(param.activation_param.has_active ? 1 : 0)};
    params_buffer_ = metal_context_->CreateBuffer(*device,
                                           &conv_params,
                                           sizeof(conv_params),
                                           METAL_ACCESS_FLAG::CPUWriteOnly);
  }
  auto filter_buffer = param.filter->template data<float>();

  if (IsWinoGrad(function_name_)) {
    //      param.filter.convert(converter: WinogradPointerConverter<P>.init())
    //      param.filter.useWinoGrad = true;
    throw std::logic_error("ERROR: still no this");
  }

  if (function_name_ == "conv_add_relu_3x3_half_winograd") {
    bool pad_when_one_ch = false;
    filter_buffer_ = make_shared<MetalBuffer>(*device,
                                              param.filter->dims(),
                                              METAL_PRECISION_TYPE::HALF,
                                              pad_when_one_ch,
                                              false,
                                              false);
  } else {
    bool pad_when_one_ch = !(param.filter->dims()[1] == 1 &&
                             param.filter->dims()[0] == param.x->dims()[1]);

    if (std::is_same<float, P>::value) {
    filter_buffer_ = make_shared<MetalBuffer>(*device,
                                              param.filter->dims(),
                                              METAL_PRECISION_TYPE::FLOAT,
                                              pad_when_one_ch,
                                              true,
                                              false);
    } else if( std::is_same<MetalHalf, P>::value) {
      filter_buffer_ = make_shared<MetalBuffer>(*device,
                                                param.filter->dims(),
                                                METAL_PRECISION_TYPE::HALF,
                                                pad_when_one_ch,
                                                true,
                                                false);
    }
  }
  filter_buffer_->CopyFromNCHW<float>(filter_buffer);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::DepthwiseConv2dImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::DepthwiseConv2dImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::DepthwiseConv2dImageCompute<float, PRECISION(kFloat)> MetalDepthwiseConv2dFp32;
typedef paddle::lite::kernels::metal::DepthwiseConv2dImageCompute<MetalHalf, PRECISION(kFP16)> MetalDepthwiseConv2dFp16;


REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalDepthwiseConv2dFp32,
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
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    MetalDepthwiseConv2dFp16,
    def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
