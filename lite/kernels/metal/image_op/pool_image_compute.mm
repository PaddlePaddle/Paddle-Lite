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

#include <cmath>

#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/kernels/metal/image_op/pool_image_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void PoolImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto device = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();
  auto global_pool = param.global_pooling;
  int pool_type;
  if (param.pooling_type == "max")
    pool_type = 0;
  else if (param.pooling_type == "avg")
    pool_type = 1;
  else {
    throw std::logic_error("ERROR: no such pooling type\n");
  }
  auto kw = param.ksize[1];
  auto kh = param.ksize[0];
  auto sw = param.strides[1];
  auto sh = param.strides[0];
  auto pw = (*param.paddings)[2];
  auto ph = (*param.paddings)[0];

  input_buffer_ = param.x->template data<P, MetalImage>();
  if (param.global_pooling) {
    kw = input_dims[3];
    kh = input_dims[2];
    auto pw = 0;
    auto ph = 0;
  }

  PoolMetalParam pool_params{kw, kh, sw, sh, pw, ph, pool_type, param.exclusive};

  params_buffer_ = metal_context_->CreateBuffer(
      *device, &pool_params, sizeof(pool_params), METAL_ACCESS_FLAG::CPUWriteOnly);

  output_buffer_ =
      param.output->template mutable_data<P, MetalImage>(output_dims, input_buffer_->transpose_);

  std::string function_name = "";
  if (std::is_same<float, P>::value) {
    function_name = "pool_float";
  } else if (std::is_same<MetalHalf, P>::value) {
    function_name = "pool_half";
  }
  assert(!function_name.empty());

  kernel_ = metal_context_->GetKernel(*device, function_name);
  queue_ = metal_context_->GetDefaultQueue(*device);
}

template <typename P, PrecisionType PTYPE>
void PoolImageCompute<P, PTYPE>::Run() {
  const auto& param = this->template Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.output->dims();

  MetalUint output_width = output_buffer_->image().width;
  MetalUint output_height = output_buffer_->image().height;
  MetalUint output_array_length = output_buffer_->image().arrayLength;

  auto encoder = std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
  MetalUint3 global_work_size = {output_width, output_height, output_array_length};

  [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
  [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
  [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

  kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::PoolImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::PoolImageCompute<MetalHalf, PRECISION(kFP16)>;
typedef paddle::lite::kernels::metal::PoolImageCompute<float, PRECISION(kFloat)> MetalPoolFp32;
typedef paddle::lite::kernels::metal::PoolImageCompute<MetalHalf, PRECISION(kFP16)> MetalPoolFp16;

REGISTER_LITE_KERNEL(pool2d,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalPoolFp32,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(pool2d,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalPoolFp16,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
