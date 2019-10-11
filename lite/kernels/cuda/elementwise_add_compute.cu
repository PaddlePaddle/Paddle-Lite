/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <vector>
#include "lite/backends/cuda/math/elementwise.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/elementwise_add_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void ElementwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const lite::Tensor* x = param.X;
  const lite::Tensor* y = param.Y;
  lite::Tensor* out = param.Out;

  CHECK(x->dims() == y->dims());

  const int n = x->dims()[0];
  const int c = x->dims()[1];
  const int h = x->dims()[2];
  const int w = x->dims()[3];

  auto* x_data = x->data<float>();
  auto* y_data = y->data<float>();
  auto out_data = out->mutable_data<float>(TARGET(kCUDA));

  int pixel_num = x->numel();
  lite::cuda::math::elementwise_add(
      pixel_num, x_data, y_data, out_data, stream);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseAddComputeNHWC::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const lite::Tensor* x = param.X;
  const lite::Tensor* y = param.Y;
  lite::Tensor* out = param.Out;

  CHECK(x->dims() == y->dims());

  const int n = x->dims()[0];
  const int c = x->dims()[1];
  const int h = x->dims()[2];
  const int w = x->dims()[3];

  auto* x_data = x->data<float>();
  auto* y_data = y->data<float>();
  auto out_data = out->mutable_data<float>(TARGET(kCUDA));

  int pixel_num = x->numel();
  lite::cuda::math::elementwise_add(
      pixel_num, x_data, y_data, out_data, stream);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseAddComputeInt8::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const lite::Tensor* x = param.X;
  const lite::Tensor* y = param.Y;
  lite::Tensor* out = param.Out;

  CHECK(x->dims() == y->dims());

  const int c = x->dims()[3];

  auto* x_data = x->data<float>();
  auto* y_data = y->data<float>();
  auto out_data = out->mutable_data<int8_t>(TARGET(kCUDA));

  int pixel_num = x->numel();
  float output_scale = param.output_scale;
  if (c % 4 == 0) {
    lite::cuda::math::elementwise_add_nhwc4_int8(
        pixel_num / 4,
        static_cast<const void*>(x_data),
        static_cast<const void*>(y_data),
        1. / output_scale,
        static_cast<void*>(out_data),
        stream);
  } else {
    lite::cuda::math::elementwise_add_int8(
        pixel_num, x_data, y_data, 1. / output_scale, out_data, stream);
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kCUDA,
                     kFloat,
                     kNHWC,
                     paddle::lite::kernels::cuda::ElementwiseAddComputeNHWC,
                     nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
