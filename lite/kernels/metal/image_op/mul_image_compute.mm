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

#include "lite/kernels/metal/image_op/mul_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/mul_image_compute.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void mul_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();

  auto s1 = 1, s2 = 1;
  for (int i = 0; i < param.x_num_col_dims; i++) {
    s1 *= input_dims[i];
  }

  for (int i = param.x_num_col_dims; i < input_dims.size(); i++) {
    s2 *= input_dims[i];
  }

  input_buffer_x_ = param.x->data<float, metal_image>();
  input_buffer_y_ = param.y->data<float, metal_image>();

  std::vector<int> nhwc = {0, 1, 2, 3};
  this->inputXMulDim = DDimLite({s1, s2});
  assert(input_buffer_y_->transpose_ == nhwc && input_buffer_y_->tensorDim_.size() == 2 &&
         s2 == input_buffer_y_->tensorDim_[0]);

  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);

  if (input_dims.size() != 2 || input_buffer_x_->transpose_ != nhwc) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->As<MetalContext>().InitOnce();
    operators::ReshapeParam reshapeParam;
    reshapeParam.x = param.x;

    shape_out_dev.Resize(this->inputXMulDim.Vectorize());
    reshapeParam.output = &shape_out_dev;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshapeParam);
    reshape_.PrepareForRun();
  }

  std::string function_name = "mul";
  kernel_ = mtl_ctx->get_kernel(*device, function_name.c_str());
}

void mul_image_compute::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.output->dims();
  auto input = param.x;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint output_width = output_buffer_->get_image().width;
    metal_uint output_height = output_buffer_->get_image().height;
    metal_uint output_array_length = output_buffer_->get_image().arrayLength;
    metal_uint3 global_work_size = {output_width, output_height, output_array_length};
    if (insert_shape) {
      reshape_.Run();
      auto shape_buffer = shape_out_dev.data<float, metal_image>();

      auto args = {metal_kernel_arg{shape_buffer},
                   metal_kernel_arg{input_buffer_y_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    } else {
      auto args = {metal_kernel_arg{input_buffer_x_},
                   metal_kernel_arg{input_buffer_y_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    }
    queue->wait_until_complete();
  }
}

void mul_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();

  auto s1 = 1, s2 = 1;
  for (int i = 0; i < param.x_num_col_dims; i++) {
    s1 *= input_dims[i];
  }

  for (int i = param.x_num_col_dims; i < input_dims.size(); i++) {
    s2 *= input_dims[i];
  }

  input_buffer_x_ = param.x->data<metal_half, metal_image>();
  input_buffer_y_ = param.y->data<metal_half, metal_image>();

  std::vector<int> nhwc = {0, 1, 2, 3};
  this->inputXMulDim = DDimLite({s1, s2});
  assert(input_buffer_y_->transpose_ == nhwc && input_buffer_y_->tensorDim_.size() == 2 &&
         s2 == input_buffer_y_->tensorDim_[0]);

  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);

  if (input_dims.size() != 2 || input_buffer_x_->transpose_ != nhwc) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->As<MetalContext>().InitOnce();
    operators::ReshapeParam reshapeParam;
    reshapeParam.x = param.x;

    shape_out_dev.Resize(this->inputXMulDim.Vectorize());
    reshapeParam.output = &shape_out_dev;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshapeParam);
    reshape_.PrepareForRun();
  }

  std::string function_name = "mul_half";
  kernel_ = mtl_ctx->get_kernel(*device, function_name.c_str());
}

void mul_image_compute_half::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.output->dims();
  auto input = param.x;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint output_width = output_buffer_->get_image().width;
    metal_uint output_height = output_buffer_->get_image().height;
    metal_uint output_array_length = output_buffer_->get_image().arrayLength;
    metal_uint3 global_work_size = {output_width, output_height, output_array_length};
    if (insert_shape) {
      reshape_.Run();
      auto shape_buffer = shape_out_dev.data<metal_half, metal_image>();
      auto args = {metal_kernel_arg{shape_buffer},
                   metal_kernel_arg{input_buffer_y_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    } else {
      auto args = {metal_kernel_arg{input_buffer_x_},
                   metal_kernel_arg{input_buffer_y_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    }
    queue->wait_until_complete();
  }
#if 0
  metal_debug::dump_image("input_buffer_x_half", input_buffer_x_, param.x->dims().production());
  metal_debug::dump_image("input_buffer_y_half", input_buffer_y_, param.y->dims().production());
  metal_debug::dump_image("output_buffer_", output_buffer_, param.output->dims().production());
#endif
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(mul,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::mul_image_compute,
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

REGISTER_LITE_KERNEL(mul,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::mul_image_compute_half,
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