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
#include "lite/kernels/metal/image_op/feed_image_compute.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void feed_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.out->dims();
  auto num = param.feed_list->size();

  auto col = param.col;

  auto input_tensor = (*param.feed_list)[col];
  auto input_buffer = input_tensor.data<float>();

  input_buffer_ = std::make_shared<metal_buffer>(*device, input_tensor.dims(), METAL_PRECISION_TYPE::FLOAT);
  output_buffer_ = param.out->mutable_data<float, metal_image>(output_dims);

  string function_name = "buffer_to_texture_array_n_channel_kernel";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}


void feed_image_compute::Run() {
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

    auto args = {metal_kernel_arg{input_buffer_}, metal_kernel_arg{output_buffer_}};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

}
}
}
}
//
//REGISTER_LITE_KERNEL(feed,
//                     kMetal,
//                     kFloat,
//                     kMetalTexture2DArray,
//                     paddle::lite::kernels::metal::feed_image_compute,
//                     def)
//        .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost),
//                                                   PRECISION(kFloat),
//                                                   DATALAYOUT(kNCHW))})
//        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
//                                                     PRECISION(kFloat),
//                                                     DATALAYOUT(kMetalTexture2DArray))})
//        .Finalize();
