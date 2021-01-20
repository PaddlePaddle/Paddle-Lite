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
#include "dropout_image_compute.h"
#include "metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void dropout_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();

  float prob_data = param.dropout_prob;
  float scale = 1.0f;
  if (param.dropout_implementation == "downgrade_in_infer") {
    scale = 1.0f - prob_data;
  }

  input_buffer_ = param.x->data<float, metal_image>();

  DropoutMetalParam metal_param{scale};

  param_buffer_ = mtl_ctx->create_buffer(*device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);
  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);

  string function_name = "dropout";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}

void dropout_image_compute::Run() {
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
            (std::pair<metal_kernel_arg, int>){param_buffer_, 0},
    };

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

    void dropout_image_compute_half::PrepareForRun() {
        auto& context = ctx_->As<MetalContext>();
        auto mtl_ctx = (metal_context*)context.context();
        auto device = mtl_ctx->get_default_device();

        const auto& param = this->Param<param_t>();
        auto output_dims = param.output->dims();

        float prob_data = param.dropout_prob;
        float scale = 1.0f;
        if (param.dropout_implementation == "downgrade_in_infer") {
            scale = 1.0f - prob_data;
        }

        input_buffer_ = param.x->data<float, metal_image>();

        DropoutMetalParam metal_param{scale};

        param_buffer_ = mtl_ctx->create_buffer(*device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);
        output_buffer_ = param.output->mutable_data<metal_half, metal_image>(output_dims);

        string function_name = "dropout_half";
        kernel_ = mtl_ctx->get_kernel(*device, function_name);
    }

    void dropout_image_compute_half::Run() {
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
                         metal_kernel_arg{param_buffer_}};

            kernel_->execute(*queue, global_work_size, 0, args);
            queue->wait_until_complete();
        }
    }

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dropout,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::dropout_image_compute,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
        .Finalize();


REGISTER_LITE_KERNEL(dropout,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::dropout_image_compute_half,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
        .Finalize();