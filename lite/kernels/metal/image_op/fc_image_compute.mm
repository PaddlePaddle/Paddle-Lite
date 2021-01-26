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

void FCImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

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

  input_buffer_ = param.input->data<float, MetalImage>();
  weight_buffer_ = param.w->data<float, MetalImage>();
  bias_buffer_ = param.bias->data<float, MetalImage>();

  std::vector<int> nhwc = {0, 1, 2, 3};
  input_x_mul_dim_ = DDimLite({s1, s2});
  assert(weight_buffer_->transpose_ == nhwc && weight_buffer_->tensor_dim_.size() == 2 &&
         s2 == weight_buffer_->tensor_dim_[0]);

  output_buffer_ = param.output->mutable_data<float, MetalImage>(output_dims);

  if (input_dims.size() != 2 || input_buffer_->transpose_ != nhwc) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->As<ContextMetal>().InitOnce();
    operators::ReshapeParam reshape_param;
    reshape_param.x = param.input;

    shape_out_dev_.Resize(input_x_mul_dim_.Vectorize());
    reshape_param.output = &shape_out_dev_;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshape_param);
    reshape_.PrepareForRun();
  }

  std::string function_name = "mul_add";
  kernel_ = mtl_ctx->GetKernel(*device, function_name.c_str());
}

void FCImageCompute::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.input->dims();
  auto output_dims = param.output->dims();
  auto input = param.input;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);
    MetalUint output_width = output_buffer_->image().width;
    MetalUint output_height = output_buffer_->image().height;
    MetalUint output_array_length = output_buffer_->image().arrayLength;
    MetalUint3 global_work_size = {output_width, output_height, output_array_length};
    if (insert_shape) {
      reshape_.Run();
      auto shape_buffer = shape_out_dev_.data<float, MetalImage>();
      auto args = {MetalKernelArgument{shape_buffer},
                   MetalKernelArgument{weight_buffer_},
                   MetalKernelArgument{bias_buffer_},
                   MetalKernelArgument{output_buffer_}};
      kernel_->Execute(*queue, global_work_size, false, args);
    } else {
      auto args = {MetalKernelArgument{input_buffer_},
                   MetalKernelArgument{weight_buffer_},
                   MetalKernelArgument{bias_buffer_},
                   MetalKernelArgument{output_buffer_}};
      kernel_->Execute(*queue, global_work_size, false, args);
    }
    queue->WaitUntilComplete();
  }
}

void FCImageComputeHalf::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

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

  input_buffer_ = param.input->data<MetalHalf, MetalImage>();
  weight_buffer_ = param.w->data<MetalHalf, MetalImage>();
  bias_buffer_ = param.bias->data<MetalHalf, MetalImage>();

  std::vector<int> nhwc = {0, 1, 2, 3};
  input_x_mul_dim_ = DDimLite({s1, s2});
  assert(weight_buffer_->transpose_ == nhwc && weight_buffer_->tensor_dim_.size() == 2 &&
         s2 == weight_buffer_->tensor_dim_[0]);

  output_buffer_ = param.output->mutable_data<MetalHalf, MetalImage>(output_dims);

  if (input_dims.size() != 2 || input_buffer_->transpose_ != nhwc) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->As<ContextMetal>().InitOnce();
    operators::ReshapeParam reshape_param;
    reshape_param.x = param.input;

    shape_out_dev_.Resize(input_x_mul_dim_.Vectorize());
    reshape_param.output = &shape_out_dev_;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshape_param);
    reshape_.PrepareForRun();
  }

  std::string function_name = "mul_add_half";
  kernel_ = mtl_ctx->GetKernel(*device, function_name.c_str());
}

void FCImageComputeHalf::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.input->dims();
  auto output_dims = param.output->dims();
  auto input = param.input;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);
    MetalUint output_width = output_buffer_->image().width;
    MetalUint output_height = output_buffer_->image().height;
    MetalUint output_array_length = output_buffer_->image().arrayLength;
    MetalUint3 global_work_size = {output_width, output_height, output_array_length};
    if (insert_shape) {
      reshape_.Run();
      auto shape_buffer = shape_out_dev_.data<float, MetalImage>();
      auto args = {MetalKernelArgument{shape_buffer},
                   MetalKernelArgument{weight_buffer_},
                   MetalKernelArgument{bias_buffer_},
                   MetalKernelArgument{output_buffer_}};

      kernel_->Execute(*queue, global_work_size, false, args);
    } else {
      auto args = {MetalKernelArgument{input_buffer_},
                   MetalKernelArgument{weight_buffer_},
                   MetalKernelArgument{bias_buffer_},
                   MetalKernelArgument{output_buffer_}};
      kernel_->Execute(*queue, global_work_size, false, args);
    }
    queue->WaitUntilComplete();
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
                     paddle::lite::kernels::metal::FCImageCompute,
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
                     paddle::lite::kernels::metal::FCImageComputeHalf,
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