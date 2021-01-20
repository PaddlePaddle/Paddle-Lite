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
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void reshape_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  transpose_ = {0, 1, 2, 3};

  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);
  input_buffer_ = param.x->data<float, metal_image>();

  int irank = input_buffer_->tensorDim_.size();
  int orank = output_buffer_->tensorDim_.size();

  std::string func_name =
      "reshape_" + std::to_string(irank) + "_" + std::to_string(orank) + "_float";
  kernel_ = mtl_ctx->get_kernel(*device, func_name);

  std::vector<int> it = input_buffer_->transpose_;
  std::vector<int> ot = output_buffer_->transpose_;
  std::vector<int> id = {1, 1, 1, 1};
  std::vector<int> od = {1, 1, 1, 1};

  for (int i = 0; i < irank; i++) {
    id[4 - irank + i] = (int)input_buffer_->tensorDim_[i];
  }

  for (int i = 0; i < orank; i++) {
    od[4 - orank + i] = (int)(output_buffer_->tensorDim_[i]);
  }

  ReshapeMetalParam reshapeMetalParam{{id[0], id[1], id[2], id[3]},
                                      {it[0], it[1], it[2], it[3]},
                                      {od[0], od[1], od[2], od[3]},
                                      {ot[0], ot[1], ot[2], ot[3]}};

  params_buffer_ = mtl_ctx->create_buffer(
      *device, &reshapeMetalParam, sizeof(reshapeMetalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
}

void reshape_image_compute::Run() {
  const auto& param = this->Param<param_t>();
  auto output = param.output;
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

    std::vector<std::pair<metal_kernel_arg, int>> args = {
            (std::pair<metal_kernel_arg, int>){input_buffer_, 0},
            (std::pair<metal_kernel_arg, int>){output_buffer_, 0},
            (std::pair<metal_kernel_arg, int>){params_buffer_, 0},
    };

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void reshape_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  transpose_ = {0, 1, 2, 3};

  output_buffer_ = param.output->mutable_data<metal_half, metal_image>(output_dims);
  input_buffer_ = param.x->data<metal_half, metal_image>();

  int irank = input_buffer_->tensorDim_.size();
  int orank = output_buffer_->tensorDim_.size();

  std::string func_name =
      "reshape_" + std::to_string(irank) + "_" + std::to_string(orank) + "_half";
  kernel_ = mtl_ctx->get_kernel(*device, func_name);

  std::vector<int> it = input_buffer_->transpose_;
  std::vector<int> ot = output_buffer_->transpose_;
  std::vector<int> id = {1, 1, 1, 1};
  std::vector<int> od = {1, 1, 1, 1};

  for (int i = 0; i < irank; i++) {
    id[4 - irank + i] = (int)input_buffer_->tensorDim_[i];
  }

  for (int i = 0; i < orank; i++) {
    od[4 - orank + i] = (int)(output_buffer_->tensorDim_[i]);
  }

  ReshapeMetalParam reshapeMetalParam{{id[0], id[1], id[2], id[3]},
                                      {it[0], it[1], it[2], it[3]},
                                      {od[0], od[1], od[2], od[3]},
                                      {ot[0], ot[1], ot[2], ot[3]}};

  params_buffer_ = mtl_ctx->create_buffer(
      *device, &reshapeMetalParam, sizeof(reshapeMetalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
}

void reshape_image_compute_half::Run() {
  const auto& param = this->Param<param_t>();
  auto output = param.output;
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

    auto args = {metal_kernel_arg{input_buffer_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{params_buffer_}};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

//REGISTER_LITE_KERNEL(reshape,
//                     kMetal,
//                     kFloat,
//                     kMetalTexture2DArray,
//                     paddle::lite::kernels::metal::reshape_image_compute,
//                     def)
//.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
//                                       PRECISION(kFloat),
//                                       DATALAYOUT(kMetalTexture2DArray))})
//    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
//                                              PRECISION(kFloat),
//                                              DATALAYOUT(kMetalTexture2DArray))})
//    .BindInput("ShapeTensor", {LiteType::GetTensorTy(TARGET(kHost),
//                                                     PRECISION(kInt32))})
//    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost),
//                                               PRECISION(kInt32))})
//    .Finalize();
//
//
//REGISTER_LITE_KERNEL(reshape,
//                     kMetal,
//                     kFP16,
//                     kMetalTexture2DArray,
//                     paddle::lite::kernels::metal::reshape_image_compute_half,
//                     def)
//.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
//                                       PRECISION(kFP16),
//                                       DATALAYOUT(kMetalTexture2DArray))})
//    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
//                                              PRECISION(kFP16),
//                                              DATALAYOUT(kMetalTexture2DArray))})
//    .BindInput("ShapeTensor", {LiteType::GetTensorTy(TARGET(kHost),
//                                                     PRECISION(kInt32))})
//    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost),
//                                               PRECISION(kInt32))})
//    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::reshape_image_compute,
    def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
PRECISION(kFloat),
DATALAYOUT(kMetalTexture2DArray))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
PRECISION(kFloat),
DATALAYOUT(kMetalTexture2DArray))})
.BindInput("ShapeTensor", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kInt32))})
.BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kInt32))})
.BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost),
                                                 PRECISION(kInt32))})
.Finalize();


REGISTER_LITE_KERNEL(reshape2,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::reshape_image_compute_half,
    def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
PRECISION(kFP16),
DATALAYOUT(kMetalTexture2DArray))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
PRECISION(kFP16),
DATALAYOUT(kMetalTexture2DArray))})
.BindInput("ShapeTensor", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kInt32))})
.BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kInt32))})
.BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost),
                                                 PRECISION(kInt32))})
.Finalize();


REGISTER_LITE_KERNEL(flatten,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::reshape_image_compute_half,
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
                     paddle::lite::kernels::metal::reshape_image_compute,
                     image2d)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                  PRECISION(kFloat),
                                  DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();


REGISTER_LITE_KERNEL(flatten2,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::reshape_image_compute_half,
                     image2d)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kMetalTexture2DArray))})
.BindInput("Shape", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kInt32))})
.BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost),
                                                 PRECISION(kInt32))})
.BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
.Finalize();


REGISTER_LITE_KERNEL(flatten2,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::reshape_image_compute,
                     image2d)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kMetalTexture2DArray))})
.BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
.BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
.Finalize();