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

#include "lite/kernels/cuda/conv_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void ConvCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  conv_impl_.reset(new lite::cuda::math::CudnnConv2D<PRECISION(kFloat)>);
  conv_impl_->init(param, &ctx);
}

void ConvCompute::Run() {
  auto& param = this->Param<param_t>();
  conv_impl_->run(param);
}

template <PrecisionType Ptype_out>
void ConvComputeInt8<Ptype_out>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  conv_impl_.reset(new lite::cuda::math::CudnnConv2DInt8<Ptype_out>);
  conv_impl_->init(param, &ctx);
}

template <PrecisionType Ptype_out>
void ConvComputeInt8<Ptype_out>::Run() {
  auto& param = this->Param<param_t>();
  conv_impl_->run(param);
}

template class ConvComputeInt8<PRECISION(kInt8)>;
template class ConvComputeInt8<PRECISION(kFloat)>;

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    conv2d, kCUDA, kFloat, kNCHW, paddle::lite::kernels::cuda::ConvCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv2d,
    kCUDA,
    kInt8,
    kNHWC,
    paddle::lite::kernels::cuda::ConvComputeInt8<PRECISION(kFloat)>,
    fp32_out)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt8))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv2d,
    kCUDA,
    kInt8,
    kNHWC,
    paddle::lite::kernels::cuda::ConvComputeInt8<PRECISION(kInt8)>,
    int8_out)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
