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
#include "lite/kernels/metal/image_op/elementwise_add_image_compute.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void elementwise_add_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();
  auto input_dims = param.X->dims();

  output_buffer_ = param.Out->mutable_data<float, metal_image>(output_dims);
  input_buffer_x_ = param.X->data<float, metal_image>();
  input_buffer_y_ = param.Y->data<float, metal_image>();

  std::vector<int> xdim, ydim, xtrans, ytrans;
  for (int i = 0; i < 4; i++) {
    xdim.push_back((int)input_buffer_x_->dim_[i]);
    ydim.push_back((int)input_buffer_y_->dim_[i]);
  }

  auto axis = param.axis;
  int params_axis = 0;
  if (axis == -1) {
    params_axis = 4 - (int)(output_buffer_->tensorDim_.size());
  } else {
    params_axis = 4 - (int)(output_buffer_->tensorDim_.size()) + axis;
  }
  int params_fast = 0;
  if ((input_buffer_x_->dim_ == input_buffer_y_->dim_) &&
      (input_buffer_x_->transpose_ == input_buffer_y_->transpose_)) {
    //      print("===> elementwise_add fast!!!")
    params_fast = 1;
  }

  int addByChannel = 0;
  if (input_buffer_y_->tensorDim_.size() == 1 &&
      (axis == 1 ||
       (axis == -1 && input_buffer_y_->tensorDim_[0] == input_buffer_x_->padToFourDim_[1]))) {
    addByChannel = 1;
  }

  ElementwiseAddMetalParam metalParam = {params_fast,
                                         addByChannel,
                                         params_axis,
                                         (int)output_buffer_->tensorDim_.size(),
                                         {xdim[0], xdim[1], xdim[2], xdim[3]},
                                         {input_buffer_x_->transpose_[0],
                                          input_buffer_x_->transpose_[1],
                                          input_buffer_x_->transpose_[2],
                                          input_buffer_x_->transpose_[3]},
                                         {ydim[0], ydim[1], ydim[2], ydim[3]},
                                         {input_buffer_y_->transpose_[0],
                                          input_buffer_y_->transpose_[1],
                                          input_buffer_y_->transpose_[2],
                                          input_buffer_y_->transpose_[3]}};

  params_buffer_ = mtl_ctx->create_buffer(*device, &metalParam, sizeof(metalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
}

void elementwise_add_image_compute::Run() {
  const auto& param = this->Param<param_t>();
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "elementwise_add";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    auto kernel = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_x_},
                 metal_kernel_arg{input_buffer_y_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{params_buffer_}};

    kernel->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void elementwise_add_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();
  auto input_dims = param.X->dims();

  output_buffer_ = param.Out->mutable_data<metal_half, metal_image>(output_dims);
  input_buffer_x_ = param.X->data<metal_half, metal_image>();
  input_buffer_y_ = param.Y->data<metal_half, metal_image>();

  std::vector<int> xdim, ydim, xtrans, ytrans;
  for (int i = 0; i < 4; i++) {
    xdim.push_back((int)input_buffer_x_->dim_[i]);
    ydim.push_back((int)input_buffer_y_->dim_[i]);
  }

  auto axis = param.axis;
  int params_axis = 0;
  if (axis == -1) {
    params_axis = 4 - (int)(output_buffer_->tensorDim_.size());
  } else {
    params_axis = 4 - (int)(output_buffer_->tensorDim_.size()) + axis;
  }
  int params_fast = 0;
  if ((input_buffer_x_->dim_ == input_buffer_y_->dim_) &&
      (input_buffer_x_->transpose_ == input_buffer_y_->transpose_)) {
    //      print("===> elementwise_add fast!!!")
    params_fast = 1;
  }

  int addByChannel = 0;
  if (input_buffer_y_->tensorDim_.size() == 1 &&
      (axis == 1 ||
       (axis == -1 && input_buffer_y_->tensorDim_[0] == input_buffer_x_->padToFourDim_[1]))) {
    addByChannel = 1;
  }

  ElementwiseAddMetalParam metalParam = {params_fast,
                                         addByChannel,
                                         params_axis,
                                         (int)output_buffer_->tensorDim_.size(),
                                         {xdim[0], xdim[1], xdim[2], xdim[3]},
                                         {input_buffer_x_->transpose_[0],
                                          input_buffer_x_->transpose_[1],
                                          input_buffer_x_->transpose_[2],
                                          input_buffer_x_->transpose_[3]},
                                         {ydim[0], ydim[1], ydim[2], ydim[3]},
                                         {input_buffer_y_->transpose_[0],
                                          input_buffer_y_->transpose_[1],
                                          input_buffer_y_->transpose_[2],
                                          input_buffer_y_->transpose_[3]}};

  params_buffer_ = mtl_ctx->create_buffer(
      *device, &metalParam, sizeof(metalParam), METAL_ACCESS_FLAG::CPUWriteOnly);
}

void elementwise_add_image_compute_half::Run() {
  const auto& param = this->Param<param_t>();
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "elementwise_add_half";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    auto kernel = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_x_},
                 metal_kernel_arg{input_buffer_y_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{params_buffer_}};

    kernel->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
#if 0
  metal_debug::dump_image("output_buffer_", output_buffer_, param.Out->dims().production());
  metal_debug::dump_image("input_buffer_x_half", input_buffer_x_, param.X->dims().production());
  metal_debug::dump_image("input_buffer_y_half", input_buffer_y_, param.Y->dims().production());
#endif
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::elementwise_add_image_compute,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("Y", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(elementwise_add,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::elementwise_add_image_compute_half,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("Y", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();