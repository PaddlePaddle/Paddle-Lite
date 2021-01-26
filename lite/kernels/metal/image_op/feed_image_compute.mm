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

void FeedImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.out->dims();
  auto num = param.feed_list->size();

  auto col = param.col;

  auto input_tensor = (*param.feed_list)[col];
  auto input_buffer = input_tensor.data<float>();

  input_buffer_ = std::make_shared<MetalBuffer>(*device, input_tensor.dims(), METAL_PRECISION_TYPE::FLOAT);
  output_buffer_ = param.out->mutable_data<float, MetalImage>(output_dims);

  string function_name = "buffer_to_texture_array_n_channel_kernel";
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}


void FeedImageCompute::Run() {
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

    auto args = {MetalKernelArgument{input_buffer_}, MetalKernelArgument{output_buffer_}};

    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
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
