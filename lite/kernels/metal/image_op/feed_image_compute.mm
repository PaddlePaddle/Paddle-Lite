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

#include "lite/kernels/metal/image_op/feed_image_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void FeedImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  device_ = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto output_dims = param.out->dims();

  Tensor& input_tensor = param.feed_list->at(param.col);
  auto input_dims = input_tensor.dims();
  int input_channel = input_dims[1];
  param.out->Resize(input_dims);
  output_buffer_ = param.out->template mutable_data<P, MetalImage>(output_dims);

  std::string function_name = "";
  if (std::is_same<float, P>::value) {
    if (input_channel == 1) {
      function_name = "buffer_to_texture_array_kernel";
    } else if (input_channel == 3) {
      function_name = "buffer_to_texture_array_kernel_channel_3";
    } else {
      function_name = "buffer_to_texture_array_n_channel_kernel";
    }
  } else if (std::is_same<MetalHalf, P>::value) {
    if (input_channel == 1) {
      function_name = "buffer_to_texture_array_kernel_half";
    } else if (input_channel == 3) {
      function_name = "buffer_to_texture_array_kernel_half_channel_3";
    } else {
      function_name = "buffer_to_texture_array_n_channel_kernel_half";
    }
  }
  assert(!function_name.empty());

  kernel_ = metal_context_->GetKernel(*device_, function_name);
  queue_ = metal_context_->GetDefaultQueue(*device_);
}

template <typename P, PrecisionType PTYPE>
void FeedImageCompute<P, PTYPE>::Run() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto mtl_dev = metal_context_->GetDefaultDevice();
  const auto& param = this->template Param<param_t>();
  Tensor& input_tensor = param.feed_list->at(param.col);
  auto input_buffer = input_tensor.mutable_data<float>();
  auto input_dims = input_tensor.dims();
  auto mem_size = input_dims.production() * sizeof(float);

  input_buffer_ = metal_context_->CreateBuffer(
      *mtl_dev, input_buffer, mem_size, METAL_ACCESS_FLAG::CPUWriteOnly);

  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto encoder = std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
  MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                 static_cast<MetalUint>(output_height),
                                 static_cast<MetalUint>(output_array_length)};

  [encoder->metal_command_encoder_ setBuffer:(input_buffer_->buffer()) offset:(0)atIndex:(0)];
  [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(0)];

  kernel_->Execute(*encoder, global_work_size, false);
}

} // namespace metal
} // namespace kernels
} // namespace lite
} // namespace paddle

template class paddle::lite::kernels::metal::FeedImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::FeedImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::FeedImageCompute<float, PRECISION(kFloat)> MetalFeedFp32;
typedef paddle::lite::kernels::metal::FeedImageCompute<MetalHalf, PRECISION(kFP16)> MetalFeedFp16;


REGISTER_LITE_KERNEL(feed,
                    kMetal,
                    kFloat,
                    kMetalTexture2DArray,
                     MetalFeedFp32,
                    def)
       .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost),
                                                  PRECISION(kFloat),
                                                  DATALAYOUT(kNCHW))})
       .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kMetalTexture2DArray))})
       .Finalize();

REGISTER_LITE_KERNEL(feed,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalFeedFp16,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kFloat),
                                               DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();
