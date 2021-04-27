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


#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/softmax_image_compute.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void SoftmaxImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto device = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();

  auto axis = param.axis;
  if (axis < 0) {
    axis += input_dims.size();
  }

  input_buffer_ = param.x->template data<P, MetalImage>();

  //  SoftmaxMetalParam metal_param{(int)input_dims[0], (int)input_dims[1]};
  SoftmaxMetalParam2 metal_param{
      (int)input_buffer_->pad_to_four_dim_[0],
      (int)input_buffer_->pad_to_four_dim_[1],
      (int)input_buffer_->pad_to_four_dim_[2],
      (int)input_buffer_->pad_to_four_dim_[3],
  };

  param_buffer_ = metal_context_->CreateBuffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);

  std::string function_name = GetFunctionName(input_dims, axis);
  assert(!function_name.empty());

  queue_ = metal_context_->GetDefaultQueue(*device);
  kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
std::string SoftmaxImageCompute<P, PTYPE>::GetFunctionName(const DDimLite& input_dims,
                                                           int axis) const {
  std::string function_name = "";
  if (std::is_same<P, float>::value) {
    if (input_dims.size() == 4) {
      if (axis == 1) {
        function_name = "softmax_c_d3_common_float";
      } else if (axis == 2) {
        function_name = "softmax_h_d3_common_float";
      } else if (axis == 3) {
        function_name = "softmax_w_d3_common_float";
      }
    }
    if (input_dims.size() == 3) {
      if (axis == 0) {
        function_name = "softmax_c_d3_common_float";
      } else if (axis == 1) {
        function_name = "softmax_h_d3_common_float";
      } else if (axis == 2) {
        function_name = "softmax_w_d3_common_float";
      }
    } else if (input_dims.size() == 2 || input_dims.size() == 1) {
      if (axis == 0) {
        function_name = "softmax_h_2d_common_float";
      } else if (axis == 1) {
        function_name = "softmax_w_2d_common_float";
      }
    } else {
      throw std::logic_error("ERROR: softmax still not support the axis");
    }
  } else if (std::is_same<MetalHalf, P>::value) {
    if (input_dims.size() == 4) {
      if (axis == 1) {
        function_name = "softmax_c_d3_common_half";
      } else if (axis == 2) {
        function_name = "softmax_h_d3_common_half";
      } else if (axis == 3) {
        function_name = "softmax_w_d3_common_half";
      }
    }
    if (input_dims.size() == 3) {
      if (axis == 0) {
        function_name = "softmax_c_d3_common_half";
      } else if (axis == 1) {
        function_name = "softmax_h_d3_common_half";
      } else if (axis == 2) {
        function_name = "softmax_w_d3_common_half";
      }
    } else if (input_dims.size() == 2 || input_dims.size() == 1) {
      if (axis == 0) {
        function_name = "softmax_h_2d_common_half";
      } else if (axis == 1) {
        function_name = "softmax_w_2d_common_half";
      }
    } else {
      throw std::logic_error("ERROR: softmax still not support the axis");
    }
  }
  return function_name;
}

template <typename P, PrecisionType PTYPE>
void SoftmaxImageCompute<P, PTYPE>::Run() {
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto encoder = std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
  MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                 static_cast<MetalUint>(output_height),
                                 static_cast<MetalUint>(output_array_length)};

  [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
  [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
  [encoder->metal_command_encoder_ setBuffer:(param_buffer_->buffer()) offset:(0)atIndex:(0)];

  kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::SoftmaxImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::SoftmaxImageCompute<MetalHalf, PRECISION(kFP16)>;
typedef paddle::lite::kernels::metal::SoftmaxImageCompute<float, PRECISION(kFloat)>
    MetalSoftmaxFp32;
typedef paddle::lite::kernels::metal::SoftmaxImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalSoftmaxFp16;

REGISTER_LITE_KERNEL(softmax,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalSoftmaxFp32,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(softmax,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalSoftmaxFp16,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();
