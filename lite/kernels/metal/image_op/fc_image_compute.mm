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

#include "lite/kernels/metal/image_op/fc_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void fc_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.input->dims();

  auto s1 = 1, s2 = 1;
  for (int i = 0; i < param.in_num_col_dims; i++) {
    s1 *= input_dims[i];
  }

  for (int i = param.in_num_col_dims; i < input_dims.size(); i++) {
    s2 *= input_dims[i];
  }

  input_buffer_ = param.input->data<float, metal_image>();
  weight_buffer_ = param.w->data<float, metal_image>();
  bias_buffer_ = param.bias->data<float, metal_image>();

  std::vector<int> nhwc = {0, 1, 2, 3};
  input_x_mul_dim_ = DDimLite({s1, s2});
  assert(weight_buffer_->transpose_ == nhwc && weight_buffer_->tensorDim_.size() == 2 &&
         s2 == weight_buffer_->tensorDim_[0]);

  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);

  if (input_dims.size() != 2 || input_buffer_->transpose_ != nhwc) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->As<MetalContext>().InitOnce();
    operators::ReshapeParam reshapeParam;
    reshapeParam.x = param.input;

    shape_out_dev_.Resize(input_x_mul_dim_.Vectorize());
    reshapeParam.output = &shape_out_dev_;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshapeParam);
    reshape_.PrepareForRun();
  }

  std::string function_name = "mul_add";
  kernel_ = mtl_ctx->get_kernel(*device, function_name.c_str());
}

void fc_image_compute::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.input->dims();
  auto output_dims = param.output->dims();
  auto input = param.input;

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
      auto shape_buffer = shape_out_dev_.data<float, metal_image>();
      auto args = {metal_kernel_arg{shape_buffer},
                   metal_kernel_arg{weight_buffer_},
                   metal_kernel_arg{bias_buffer_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    } else {
      auto args = {metal_kernel_arg{input_buffer_},
                   metal_kernel_arg{weight_buffer_},
                   metal_kernel_arg{bias_buffer_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    }
    queue->wait_until_complete();
  }
}

void fc_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.input->dims();

  auto s1 = 1, s2 = 1;
  for (int i = 0; i < param.in_num_col_dims; i++) {
    s1 *= input_dims[i];
  }

  for (int i = param.in_num_col_dims; i < input_dims.size(); i++) {
    s2 *= input_dims[i];
  }

  input_buffer_ = param.input->data<metal_half, metal_image>();
  weight_buffer_ = param.w->data<metal_half, metal_image>();
  bias_buffer_ = param.bias->data<metal_half, metal_image>();

  std::vector<int> nhwc = {0, 1, 2, 3};
  input_x_mul_dim_ = DDimLite({s1, s2});
  assert(weight_buffer_->transpose_ == nhwc && weight_buffer_->tensorDim_.size() == 2 &&
         s2 == weight_buffer_->tensorDim_[0]);

  output_buffer_ = param.output->mutable_data<metal_half, metal_image>(output_dims);

  if (input_dims.size() != 2 || input_buffer_->transpose_ != nhwc) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->As<MetalContext>().InitOnce();
    operators::ReshapeParam reshapeParam;
    reshapeParam.x = param.input;

    shape_out_dev_.Resize(input_x_mul_dim_.Vectorize());
    reshapeParam.output = &shape_out_dev_;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshapeParam);
    reshape_.PrepareForRun();
  }

  std::string function_name = "mul_add_half";
  kernel_ = mtl_ctx->get_kernel(*device, function_name.c_str());
}

void fc_image_compute_half::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.input->dims();
  auto output_dims = param.output->dims();
  auto input = param.input;

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
      auto shape_buffer = shape_out_dev_.data<float, metal_image>();
      auto args = {metal_kernel_arg{shape_buffer},
                   metal_kernel_arg{weight_buffer_},
                   metal_kernel_arg{bias_buffer_},
                   metal_kernel_arg{output_buffer_}};

      kernel_->execute(*queue, global_work_size, 0, args);
    } else {
      auto args = {metal_kernel_arg{input_buffer_},
                   metal_kernel_arg{weight_buffer_},
                   metal_kernel_arg{bias_buffer_},
                   metal_kernel_arg{output_buffer_}};
      kernel_->execute(*queue, global_work_size, 0, args);
    }
    queue->wait_until_complete();
  }
}

}
}
}
}

REGISTER_LITE_KERNEL(fc,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::fc_image_compute,
                     def)
        .BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kMetal),
                                                PRECISION(kFloat),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("W", {LiteType::GetTensorTy(TARGET(kMetal),
                         PRECISION(kFloat),
                         DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();

REGISTER_LITE_KERNEL(fc,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::fc_image_compute_half,
                     def)
.BindInput("Input", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kMetal),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();