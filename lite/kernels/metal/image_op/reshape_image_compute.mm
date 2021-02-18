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

void ReshapeImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  transpose_ = {0, 1, 2, 3};

  output_buffer_ = param.output->mutable_data<float, MetalImage>(output_dims);
  input_buffer_ = param.x->data<float, MetalImage>();

  int irank = input_buffer_->tensor_dim_.size();
  int orank = output_buffer_->tensor_dim_.size();

  std::string func_name = "reshape_" + std::to_string(irank) + "_" +
                          std::to_string(orank) + "_float";
  kernel_ = mtl_ctx->GetKernel(*device, func_name);

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

  params_buffer_ = mtl_ctx->CreateBuffer(*device,
                                         &reshape_params,
                                         sizeof(reshape_params),
                                         METAL_ACCESS_FLAG::CPUWriteOnly);
}

void ReshapeImageCompute::Run() {
  const auto& param = this->Param<param_t>();
  auto output = param.output;
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);

    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    std::vector<std::pair<MetalKernelArgument, int>> args = {
        (std::pair<MetalKernelArgument, int>){input_buffer_, 0},
        (std::pair<MetalKernelArgument, int>){output_buffer_, 0},
        (std::pair<MetalKernelArgument, int>){params_buffer_, 0},
    };

    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("reshape", output_buffer_);
#endif
}

void ReshapeImageComputeHalf::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  transpose_ = {0, 1, 2, 3};

  output_buffer_ =
      param.output->mutable_data<MetalHalf, MetalImage>(output_dims);
  input_buffer_ = param.x->data<MetalHalf, MetalImage>();

  int irank = input_buffer_->tensor_dim_.size();
  int orank = output_buffer_->tensor_dim_.size();

  std::string func_name = "reshape_" + std::to_string(irank) + "_" +
                          std::to_string(orank) + "_half";
  kernel_ = mtl_ctx->GetKernel(*device, func_name);

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

  params_buffer_ = mtl_ctx->CreateBuffer(*device,
                                         &reshape_params,
                                         sizeof(reshape_params),
                                         METAL_ACCESS_FLAG::CPUWriteOnly);
}

void ReshapeImageComputeHalf::Run() {
  const auto& param = this->Param<param_t>();
  auto output = param.output;
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);

    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    auto args = {MetalKernelArgument{input_buffer_},
                 MetalKernelArgument{output_buffer_},
                 MetalKernelArgument{params_buffer_}};

    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("reshape", output_buffer_);
#endif
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reshape2,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::ReshapeImageCompute,
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
                     paddle::lite::kernels::metal::ReshapeImageComputeHalf,
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
                     paddle::lite::kernels::metal::ReshapeImageComputeHalf,
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
                     paddle::lite::kernels::metal::ReshapeImageCompute,
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
                     paddle::lite::kernels::metal::ReshapeImageComputeHalf,
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
                     paddle::lite::kernels::metal::ReshapeImageCompute,
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