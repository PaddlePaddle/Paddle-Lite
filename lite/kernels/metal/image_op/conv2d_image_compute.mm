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

#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/conv2d_image_compute.h"

#define LZY_DEBUG 0
using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void conv2d_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();
  input_buffer_ = param.x->data<float, metal_image>();
  if (param.bias) bias_buffer_ = param.bias->data<float, metal_image>();

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

  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);

  float* blank_host = (float*)malloc(sizeof(float) * output_dims[1]);
  memset(blank_host, 0, sizeof(float) * output_dims[1]);

  DDim blank_dim = DDimLite({output_dims[1]});
  blank_tensor_.Resize(blank_dim);
  blank_tensor_.mutable_data<float, metal_image>(blank_dim, {0, 1, 2, 3}, (void*)blank_host);
  free(blank_host);
  blank_host = nullptr;

  bool shouldUseMPS = false;
  function_name_ = kernelFunctionName(param, mtl_ctx->get_use_aggressive_optimization());

#ifdef TARGET_IOS
    if(@available(iOS 11.0, *) {
#endif
    if (mtl_ctx->get_use_mps() || mtl_ctx->get_use_aggressive_optimization()) {
      if (input_dims[1] >= 3 && output_buffer_->tensorDim_[1] >= 3) {
        // shouldUseMPS = true; //TODO: add MPS support
      }
    }
#ifdef TARGET_IOS
    }
#endif
  if (isWinoGrad(function_name_)) {
    shouldUseMPS = false;
  }

  int filter_channel = static_cast<int>(param.filter->dims()[1]);
  int filter_n = static_cast<int>(param.filter->dims()[0]);
  bool isDepthWise = filter_channel == 1 && filter_n == input_buffer_->tensorDim_[1];
  if (!isDepthWise && param.groups > 1) {
    shouldUseMPS = false;
  }

  if (function_name_ == "") {
    throw std::logic_error("ERROR: cannot find the name of the conv2d");
  }

  if (activate_type_ == 2) {
    auto index = function_name_.find("relu");
    if (index != -1) function_name_.replace(index, 4, "relu6");
  }

  program_ = mtl_ctx->get_kernel(*device, function_name_);

  if (shouldUseMPS) {
    setupWithMPS();
  } else {
    setupWithoutMPS();
  }
}

void conv2d_image_compute::Run() {
  const auto& param = this->Param<param_t>();
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    if (param.bias) {
      std::vector<metal_kernel_arg> args {
          metal_kernel_arg{input_buffer_},
          metal_kernel_arg{bias_buffer_},
          metal_kernel_arg{output_buffer_},
          metal_kernel_arg{params_buffer_},
          metal_kernel_arg{filter_buffer_}
      };
      bool quadruple = false;
      if (isWinoGrad(function_name_) || function_name_ == "conv_add_relu_1x1_quadruple_half") {
        quadruple = true;
      }
      program_->execute(*queue, global_work_size, quadruple, args);
      queue->wait_until_complete();
    } else {
      auto blank_buffer = blank_tensor_.data<float, metal_image>();
      std::vector<metal_kernel_arg> args {
          metal_kernel_arg{input_buffer_},
          metal_kernel_arg{blank_buffer},
          metal_kernel_arg{output_buffer_},
          metal_kernel_arg{params_buffer_},
          metal_kernel_arg{filter_buffer_}
      };

      bool quadruple = false;
      if (isWinoGrad(function_name_) || function_name_ == "conv_add_relu_1x1_quadruple_half") {
        quadruple = true;
      }
      program_->execute(*queue, global_work_size, quadruple, args);
      queue->wait_until_complete();
    }
  }
}

string conv2d_image_compute::kernelFunctionName(const param_t& param,
                                                bool useAggressiveOptimization) {
  auto filter_width = param.filter->dims()[3];
  auto filter_height = param.filter->dims()[2];
  auto filter_channel = param.filter->dims()[1];
  auto filter_n = param.filter->dims()[0];
  auto padLeft = (*param.paddings)[2];
  auto padTop = (*param.paddings)[0];

  auto input_tensor_dim = param.x->dims();
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
  } else if (filter_width == 1 && filter_height == 5) {
    return "conv_add_relu_5x1";
  } else if (filter_width == 5 && filter_height == 1) {
    return "conv_add_relu_1x5";
  } else if (filter_width == 7 && filter_height == 7) {
    return "conv_add_relu_7x7";
  } else {
    return "";
  }
}

bool conv2d_image_compute::isWinoGrad(string function_name) {
  std::string suffix = "winograd";
  if (function_name.size() >= suffix.size() &&
      function_name.compare(function_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return true;
  }
  return false;
}

void conv2d_image_compute::setupWithMPS() {
  // TODO: add MPS support
}

void conv2d_image_compute::setupWithoutMPS() {
  const auto& param = this->Param<param_t>();
  auto padLeft = (*param.paddings)[2];
  auto padTop = (*param.paddings)[0];
  assert((*param.paddings)[0] == (*param.paddings)[1]);

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  int offsetX = static_cast<int>(
      ((int)((*param.dilations)[1]) * (param.filter->dims()[3] - 1) + 1) / 2 - padLeft);
  int offsetY = static_cast<int>(
      ((int)((*param.dilations)[0]) * (param.filter->dims()[2] - 1) + 1) / 2 - padTop);

  float offsetZ = 0.0;
  int iC = static_cast<int>(param.x->dims()[1]);
  int fC = static_cast<int>(param.filter->dims()[1]);
  int oC = static_cast<int>(param.output->dims()[1]);

  if (param.bias) {
    int xdim[4], ydim[4], xtrans[4], ytrans[4];
    for (int i = 0; i < 4; i++) {
      xdim[i] = (int)output_buffer_->dim_[i];
      ydim[i] = (int)bias_buffer_->dim_[i];
    }

    int axis = -1;
    int params_axis;
    if (axis == -1) {
      params_axis = 4 - (int)(output_buffer_->tensorDim_.size());
    } else {
      params_axis = 4 - (int)(output_buffer_->tensorDim_.size()) + axis;
    }

    int params_fast = 0;
    if ((output_buffer_->dim_ == bias_buffer_->dim_) &&
        (output_buffer_->transpose_ == bias_buffer_->transpose_)) {
      //      print("===> elementwise_add fast!!!")
      params_fast = 1;
    }

    int addByChannel = 0;
    if (bias_buffer_->tensorDim_.size() == 1 &&
        (axis == 1 ||
         (axis == -1 && bias_buffer_->tensorDim_[0] == output_buffer_->padToFourDim_[1]))) {
      addByChannel = 1;
    }

    ElementwiseAddMetalParam metalParam = {params_fast,
                                           addByChannel,
                                           params_axis,
                                           (int)output_buffer_->tensorDim_.size(),
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

    MetalConvParam inMetalParam{(short)offsetX,
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
                                metalParam};

    params_buffer_ = mtl_ctx->create_buffer(
        *device, &inMetalParam, sizeof(inMetalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
  } else {
    MetalConvParam inMetalParam{(short)offsetX,
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
    params_buffer_ = mtl_ctx->create_buffer(
        *device, &inMetalParam, sizeof(inMetalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
  }
  auto filter_buffer = param.filter->data<float>();

  if (isWinoGrad(function_name_)) {
    data_converter<float>* converter = new WinogradPointerConverter<float>();
    free(converter);
    std::logic_error("ERROR: still not finish winograd");
  }

  if (function_name_ == "conv_add_relu_3x3_half_winograd") {
    bool padWhenOneC = false;
    filter_buffer_ = make_shared<metal_buffer>(
        *device, param.filter->dims(), METAL_PRECISION_TYPE::HALF, padWhenOneC, false, false);
  } else {
    bool padWhenOneC =
        !(param.filter->dims()[1] == 1 && param.filter->dims()[0] == param.x->dims()[1]);
    filter_buffer_ = make_shared<metal_buffer>(
        *device, param.filter->dims(), METAL_PRECISION_TYPE::FLOAT, padWhenOneC, true, false);
  }
  filter_buffer_->from_nchw<float>(filter_buffer);
}

void conv2d_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();
  input_buffer_ = param.x->data<metal_half, metal_image>();
  if (param.bias) bias_buffer_ = param.bias->data<metal_half, metal_image>();

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

  output_buffer_ = param.output->mutable_data<metal_half, metal_image>(output_dims);

  metal_half* blank_host = (metal_half*)malloc(sizeof(metal_half) * output_dims[1]);
  memset(blank_host, 0, sizeof(metal_half) * output_dims[1]);

  DDim blank_dim = DDimLite({output_dims[1]});
  blank_tensor_.Resize(blank_dim);
  blank_tensor_.mutable_data<metal_half, metal_image>(blank_dim, {0, 1, 2, 3}, (void*)blank_host);
  free(blank_host);
  blank_host = nullptr;

  bool shouldUseMPS = false;
  function_name_ = kernelFunctionName(param, mtl_ctx->get_use_aggressive_optimization());

#ifdef TARGET_IOS
    if(@available(iOS 11.0, *) {
#endif
    if (mtl_ctx->get_use_mps() || mtl_ctx->get_use_aggressive_optimization()) {
      if (input_dims[1] >= 3 && output_buffer_->tensorDim_[1] >= 3) {
        shouldUseMPS = true;
      }
    }
#ifdef TARGET_IOS
    }
#endif
  if (isWinoGrad(function_name_)) {
    shouldUseMPS = false;
  }

  int filter_channel = static_cast<int>(param.filter->dims()[1]);
  int filter_n = static_cast<int>(param.filter->dims()[0]);
  bool isDepthWise = filter_channel == 1 && filter_n == input_buffer_->tensorDim_[1];
  if (!isDepthWise && param.groups > 1) {
    shouldUseMPS = false;
  }

  if (function_name_ == "") {
    throw std::logic_error("ERROR: cannot find the name of the conv2d");
  }

  if (activate_type_ == 2) {
    auto index = function_name_.find("relu");
    if (index != -1) function_name_.replace(index, 4, "relu6");
  }

  program_ = mtl_ctx->get_kernel(*device, function_name_);

  if (shouldUseMPS) {
    setupWithMPS();
  } else {
    setupWithoutMPS();
  }
}

void conv2d_image_compute_half::Run() {
  const auto& param = this->Param<param_t>();
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    if (param.bias) {
      std::vector<metal_kernel_arg> args = {
              metal_kernel_arg{input_buffer_},
              metal_kernel_arg{bias_buffer_},
              metal_kernel_arg{output_buffer_},
              metal_kernel_arg{params_buffer_},
              metal_kernel_arg{filter_buffer_}
      };
      bool quadruple = false;
      if (isWinoGrad(function_name_) || function_name_ == "conv_add_relu_1x1_quadruple_half") {
        quadruple = true;
      }
      program_->execute(*queue, global_work_size, quadruple, args);
      queue->wait_until_complete();
    } else {
      auto blank_buffer = blank_tensor_.data<metal_half, metal_image>();

      std::vector<metal_kernel_arg> args {
          metal_kernel_arg(input_buffer_),
          metal_kernel_arg(blank_buffer),
          metal_kernel_arg(output_buffer_),
          metal_kernel_arg(params_buffer_),
          metal_kernel_arg(filter_buffer_)
      };

      bool quadruple = false;
      if (isWinoGrad(function_name_) || function_name_ == "conv_add_relu_1x1_quadruple_half") {
        quadruple = true;
      }
      program_->execute(*queue, global_work_size, quadruple, args);
      queue->wait_until_complete();
    }
  }
}

string conv2d_image_compute_half::kernelFunctionName(const param_t& param,
                                                     bool useAggressiveOptimization) {
  auto filter_width = param.filter->dims()[3];
  auto filter_height = param.filter->dims()[2];
  auto filter_channel = param.filter->dims()[1];
  auto filter_n = param.filter->dims()[0];
  auto padLeft = (*param.paddings)[2];
  auto padTop = (*param.paddings)[0];
  auto dilations = (*param.dilations);

  auto input_tensor_dim = param.x->dims();
  if (filter_width == 1 && filter_height == 1) {
    if (filter_channel <= 16 && padLeft == 0 && padTop == 0) {
      return "conv_add_relu_1x1_quadruple_half";
    } else {
      return "conv_add_relu_1x1_half";
    }
  } else if (filter_width == 3 && filter_height == 3) {
    if (filter_channel == 1 && param.filter->dims()[0] == param.x->dims()[1]) {
      if (useAggressiveOptimization) {
        bool couldUseWinograd = filter_width == 3 && filter_height == 3 && param.strides[0] == 1 &&
                                param.strides[1] == 1 && dilations[0] == 1 && dilations[1] == 1 &&
                                padLeft == 1 && padTop == 1;
        if (couldUseWinograd) {
          return "depthwise_conv_add_relu_3x3_half_winograd";
        }
      }
      return "depthwise_conv_add_relu_3x3_half";
    } else {
      if (param.groups == 1) {
        if (useAggressiveOptimization) {
          bool couldUseWinograd = filter_width == 3 && filter_height == 3 &&
                                  param.strides[0] == 1 && param.strides[1] == 1 &&
                                  dilations[0] == 1 && dilations[1] == 1 && padLeft == 1 &&
                                  padTop == 1;
          if (couldUseWinograd) {
            return "conv_add_relu_3x3_half_winograd";
          }
        }
        return "conv_add_relu_3x3_half";
      } else {
        return "group_conv_add_relu_3x3_half";
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

bool conv2d_image_compute_half::isWinoGrad(string function_name) {
  std::string suffix = "winograd";
  if (function_name.size() >= suffix.size() &&
      function_name.compare(function_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return true;
  }
  return false;
}

void conv2d_image_compute_half::setupWithMPS() {
  // TODO: add MPS support
}

void conv2d_image_compute_half::setupWithoutMPS() {
  const auto& param = this->Param<param_t>();
  auto padLeft = (*param.paddings)[2];
  auto padTop = (*param.paddings)[0];
  assert((*param.paddings)[0] == (*param.paddings)[1]);

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  int offsetX = static_cast<int>(
      ((int)((*param.dilations)[1]) * (param.filter->dims()[3] - 1) + 1) / 2 - padLeft);
  int offsetY = static_cast<int>(
      ((int)((*param.dilations)[0]) * (param.filter->dims()[2] - 1) + 1) / 2 - padTop);

  float offsetZ = 0.0;
  int iC = static_cast<int>(param.x->dims()[1]);
  int fC = static_cast<int>(param.filter->dims()[1]);
  int oC = static_cast<int>(param.output->dims()[1]);

  if (param.bias) {
    int xdim[4], ydim[4], xtrans[4], ytrans[4];
    for (int i = 0; i < 4; i++) {
      xdim[i] = (int)output_buffer_->dim_[i];
      ydim[i] = (int)bias_buffer_->dim_[i];
    }

    int axis = -1;
    int params_axis;
    if (axis == -1) {
      params_axis = 4 - (int)(output_buffer_->tensorDim_.size());
    } else {
      params_axis = 4 - (int)(output_buffer_->tensorDim_.size()) + axis;
    }

    int params_fast = 0;
    if ((output_buffer_->dim_ == bias_buffer_->dim_) &&
        (output_buffer_->transpose_ == bias_buffer_->transpose_)) {
      //      print("===> elementwise_add fast!!!")
      params_fast = 1;
    }

    int addByChannel = 0;
    if (bias_buffer_->tensorDim_.size() == 1 &&
        (axis == 1 ||
         (axis == -1 && bias_buffer_->tensorDim_[0] == output_buffer_->padToFourDim_[1]))) {
      addByChannel = 1;
    }

    ElementwiseAddMetalParam metalParam = {params_fast,
                                           addByChannel,
                                           params_axis,
                                           (int)output_buffer_->tensorDim_.size(),
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

    MetalConvParam inMetalParam{(short)offsetX,
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
                                metalParam};

    params_buffer_ = mtl_ctx->create_buffer(
        *device, &inMetalParam, sizeof(inMetalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
  } else {
    MetalConvParam inMetalParam{(short)offsetX,
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
    params_buffer_ = mtl_ctx->create_buffer(
        *device, &inMetalParam, sizeof(inMetalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
  }
  auto filter_buffer = param.filter->data<float>();

  if (isWinoGrad(function_name_)) {
    data_converter<float>* converter = new WinogradPointerConverter<float>();
    free(converter);
    std::logic_error("ERROR: still not finish winograd");
  }

  if (function_name_ == "conv_add_relu_3x3_half_winograd") {
    bool padWhenOneC = false;
    filter_buffer_ = make_shared<metal_buffer>(
        *device, param.filter->dims(), METAL_PRECISION_TYPE::HALF, padWhenOneC, false, false);
  } else {
    bool padWhenOneC =
        !(param.filter->dims()[1] == 1 && param.filter->dims()[0] == param.x->dims()[1]);
    filter_buffer_ = make_shared<metal_buffer>(
        *device, param.filter->dims(), METAL_PRECISION_TYPE::HALF, padWhenOneC, true, false);
  }
  filter_buffer_->from_nchw<float>(filter_buffer);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::conv2d_image_compute,
                     def)
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kMetalTexture2DArray))})
.BindInput("Bias", {LiteType::GetTensorTy(TARGET(kMetal),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kMetalTexture2DArray))})
.BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost),
                                                PRECISION(kFloat),
                                                DATALAYOUT(kNCHW))})
.BindOutput("Output", {LiteType::GetTensorTy(TARGET(kMetal),
                                                 PRECISION(kFloat),
                                                 DATALAYOUT(kMetalTexture2DArray))})
.Finalize();


REGISTER_LITE_KERNEL(conv2d,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::conv2d_image_compute_half,
                     def)
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kMetalTexture2DArray))})
.BindInput("Bias", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
.BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kNCHW))})
.BindOutput("Output", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFP16),
                                                     DATALAYOUT(kMetalTexture2DArray))})
.Finalize();

