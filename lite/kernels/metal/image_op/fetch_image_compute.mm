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

#include "lite/kernels/metal/image_op/fetch_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void FetchImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  device_ = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto input_dims = param.input->dims();
  input_buffer_ = param.input->template data<P, MetalImage>();

  auto* fetch_list = param.fetch_list;
  if (fetch_list->size() <= static_cast<size_t>(param.col)) {
    fetch_list->resize(static_cast<unsigned long>(param.col + 1));
  }

  auto& dst = fetch_list->at(param.col);
  dst.Resize(param.input->dims());

  std::string function_name = "";
  if (std::is_same<P, MetalHalf>::value) {
    if (input_buffer_->transpose_ == std::vector<int>{0, 2, 3, 1}) {
      function_name = "fetch_half";
    } else if (input_buffer_->transpose_ == std::vector<int>{0, 1, 2, 3}) {
      switch (input_buffer_->tensor_dim_.size()) {
        case 1:
        case 2:
          function_name = "fetch_1or2_half";
          break;
        case 4:
          expected_transpose_ = {0, 2, 3, 1};
          function_name = "fetch_half";
          break;
        default:
          throw std::logic_error("ERROR: half compute unsupported tensor dim count");
      }
    }
  } else if (std::is_same<P, float>::value) {
    if (input_buffer_->transpose_ == std::vector<int>{0, 2, 3, 1}) {
      function_name = "fetch_float";
    } else if (input_buffer_->transpose_ == std::vector<int>{0, 1, 2, 3}) {
      switch (input_buffer_->tensor_dim_.size()) {
        case 1:
        case 2:
          function_name = "fetch_1or2_float";
          break;
        case 4:
          expected_transpose_ = {0, 2, 3, 1};
          function_name = "fetch_float";
          break;
        default:
          throw std::logic_error("ERROR: float compute unsupported tensor dim count");
      }
    }
  } else {
    throw std::logic_error("ERROR: unsupported compute precision");
  }

  if (input_buffer_->transpose_ != expected_transpose_) {
    insert_shape = true;
    std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
    reshape_ctx->template As<ContextMetal>().InitOnce();
    operators::ReshapeParam reshape_param;
    reshape_param.x = param.input;
    reshape_param.excepted_transpose_ = expected_transpose_;
    shape_out_dev.Resize(input_buffer_->tensor_dim_);
    reshape_param.output = &shape_out_dev;
    reshape_.SetContext(std::move(reshape_ctx));
    reshape_.SetParam(reshape_param);
    reshape_.PrepareForRun();
  }

  assert(!function_name.empty());
  kernel_ = metal_context_->GetKernel(*device_, function_name);
  queue_ = metal_context_->GetDefaultQueue(*device_);
}

template <typename P, PrecisionType PTYPE>
void FetchImageCompute<P, PTYPE>::Run() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto mtl_dev = metal_context_->GetDefaultDevice();
  const auto& param = this->template Param<param_t>();
  Tensor& output_tensor = param.fetch_list->at(param.col);
  auto output_buffer = output_tensor.mutable_data<float>();
  auto output_dims = output_tensor.dims();
  auto mem_size = output_dims.production() * sizeof(float);
  output_buffer_ = metal_context_->CreateBuffer(*mtl_dev, mem_size);

  auto output_width = input_buffer_->texture_width_;
  auto output_height = input_buffer_->texture_height_;
  auto output_array_length = input_buffer_->array_length_;

  MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                 static_cast<MetalUint>(output_height),
                                 static_cast<MetalUint>(output_array_length)};

  if (insert_shape) {
    reshape_.Run();
    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    auto shape_buffer = shape_out_dev.data<P, MetalImage>();
    [encoder->metal_command_encoder_ setTexture:shape_buffer->image() atIndex:(0)];
    [encoder->metal_command_encoder_ setBuffer:output_buffer_->buffer() offset:(0)atIndex:(0)];
    kernel_->Execute(*encoder, global_work_size, false);
  } else {
    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    [encoder->metal_command_encoder_ setTexture:input_buffer_->image() atIndex:(0)];
    [encoder->metal_command_encoder_ setBuffer:output_buffer_->buffer() offset:(0)atIndex:(0)];
    kernel_->Execute(*encoder, global_work_size, false);
  }

  [metal_context_->cmd_buf_->metal_command_buffer_ commit];
  [metal_context_->cmd_buf_->metal_command_buffer_ waitUntilCompleted];
  metal_context_->cmd_buf_->have_command_ = false;

  memcpy(output_buffer, output_buffer_->buffer().contents, mem_size);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::FetchImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::FetchImageCompute<MetalHalf, PRECISION(kFP16)>;
typedef paddle::lite::kernels::metal::FetchImageCompute<float, PRECISION(kFloat)> MetalFetchFp32;
typedef paddle::lite::kernels::metal::FetchImageCompute<MetalHalf, PRECISION(kFP16)> MetalFetchFp16;

REGISTER_LITE_KERNEL(fetch,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalFetchFp32,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW)
                                              )})
    .Finalize();

REGISTER_LITE_KERNEL(fetch,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalFetchFp16,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW)
                                          )})
    .Finalize();