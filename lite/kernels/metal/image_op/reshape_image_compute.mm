// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/metal/image_op/reshape_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/backends/metal/metal_debug.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {


template <typename P, PrecisionType PTYPE>
void ReshapeImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto device = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto output_dims = param.output->dims();
  auto transpose = param.excepted_transpose_;

  if(transpose.empty()){
      output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);
  } else {
    output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims, transpose);
  }
  input_buffer_ = param.x->template data<P, MetalImage>();

  auto irank = input_buffer_->tensor_dim_.size();
  auto orank = output_buffer_->tensor_dim_.size();

  std::string function_name = "";
  if (std::is_same<P, float>::value) {
    function_name = "reshape_" + std::to_string(irank) + "_" + std::to_string(orank) + "_float";
  } else if (std::is_same<P, MetalHalf>::value) {
    function_name = "reshape_" + std::to_string(irank) + "_" + std::to_string(orank) + "_half";
  }
  assert(!function_name.empty());

  kernel_ = metal_context_->GetKernel(*device, function_name);

  std::vector<int> it = input_buffer_->transpose_;
  std::vector<int> ot = output_buffer_->transpose_;
  std::vector<int> id = {1, 1, 1, 1};
  std::vector<int> od = {1, 1, 1, 1};

  for (int i = 0; i < irank; i++) {
    id[4 - irank + i] = (int)input_buffer_->tensor_dim_[i];
  }

  for (int i = 0; i < orank; i++) {
    od[4 - orank + i] = (int)(output_buffer_->tensor_dim_[i]);
  }

  ReshapeMetalParam reshape_params{{id[0], id[1], id[2], id[3]},
                                   {it[0], it[1], it[2], it[3]},
                                   {od[0], od[1], od[2], od[3]},
                                   {ot[0], ot[1], ot[2], ot[3]}};

  params_buffer_ = metal_context_->CreateBuffer(*device,
                                         &reshape_params,
                                         sizeof(reshape_params),
                                         METAL_ACCESS_FLAG::CPUWriteOnly);
  queue_ = metal_context_->GetDefaultQueue(*device);

}

template<typename P, PrecisionType PTYPE>
void ReshapeImageCompute<P, PTYPE>::Run() {
  const auto& param = this->template Param<param_t>();
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto encoder = std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
  MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                 static_cast<MetalUint>(output_height),
                                 static_cast<MetalUint>(output_array_length)};

  [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
  [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
  [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0)atIndex:(0)];

  kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::ReshapeImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::ReshapeImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::ReshapeImageCompute<float, PRECISION(kFloat)> MetalReshapeFp32;
typedef paddle::lite::kernels::metal::ReshapeImageCompute<MetalHalf, PRECISION(kFP16)> MetalReshapeFp16;

REGISTER_LITE_KERNEL(reshape2,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalReshapeFp32,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalReshapeFp16,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalReshapeFp16,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalReshapeFp32,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalReshapeFp16,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalReshapeFp32,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();