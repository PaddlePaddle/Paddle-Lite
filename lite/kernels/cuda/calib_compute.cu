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

#include <vector>

#include "lite/backends/cuda/math/utils.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/cuda/calib_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

__global__ void Fp32ToInt8Kernel(const int num,
                                 const float scale,
                                 const float* input,
                                 int8_t* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = lite::cuda::math::from_float<int8_t>(input[index] / scale);
  }
}

__global__ void Int8ToFp32Kernel(const int num,
                                 const float scale,
                                 const int8_t* input,
                                 float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = input[index] * scale;
  }
}

__global__ void Fp32ToFp16Kernel(const int num,
                                 const float* input,
                                 half* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = __float2half(input[index]);
  }
}

__global__ void Fp16ToFp32Kernel(const int num,
                                 const half* input,
                                 float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = lite::cuda::math::from_float<half>(input[index]);
  }
}

void CalibComputeFp32ToInt8::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto scale = param.scale;
  const auto* din = param.input->data<float>();
  auto* dout = param.output->mutable_data<int8_t>(TARGET(kCUDA));
  int num = static_cast<int>(param.input->numel());
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  Fp32ToInt8Kernel<<<blocks, threads, 0, stream>>>(num, scale, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void CalibComputeInt8ToFp32::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto scale = param.scale;
  const auto* din = param.input->data<int8_t>();
  auto* dout = param.output->mutable_data<float>(TARGET(kCUDA));
  int num = static_cast<int>(param.input->numel());
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  Int8ToFp32Kernel<<<blocks, threads, 0, stream>>>(num, scale, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void CalibComputeFp32ToFp16::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const auto* din = param.input->data<float>();
  auto* dout = param.output->mutable_data<__half>(TARGET(kCUDA));
  int num = static_cast<int>(param.input->numel());
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  param.output->set_lod(param.input->lod());
  Fp32ToFp16Kernel<<<blocks, threads, 0, stream>>>(num, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void CalibOnceComputeFp32ToFp16::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const auto* din = param.input->data<float>();
  auto* dout = param.output->mutable_data<__half>(TARGET(kCUDA));
  int num = static_cast<int>(param.input->numel());
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  param.output->set_lod(param.input->lod());
  Fp32ToFp16Kernel<<<blocks, threads>>>(num, din, dout);

  // remove the unneeded fp32 weights.
  const_cast<lite::Tensor*>(param.input)->clear();

  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void CalibComputeFp16ToFp32::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const auto* din = param.input->data<__half>();
  auto* dout = param.output->mutable_data<float>(TARGET(kCUDA));
  int num = static_cast<int>(param.input->numel());
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  param.output->set_lod(param.input->lod());
  Fp16ToFp32Kernel<<<blocks, threads, 0, stream>>>(num, din, dout);
  cudaError_t error = cudaGetLastError();
  CHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(calib,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeFp32ToInt8,
                     fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeInt8ToFp32,
                     int8_to_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeFp16ToFp32,
                     fp16_to_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
REGISTER_LITE_KERNEL(calib,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeFp32ToFp16,
                     fp32_to_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeFp32ToInt8,
                     fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kAny))})
    .Finalize();
REGISTER_LITE_KERNEL(calib_once,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeInt8ToFp32,
                     int8_to_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibComputeFp16ToFp32,
                     fp16_to_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
REGISTER_LITE_KERNEL(calib_once,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::CalibOnceComputeFp32ToFp16,
                     fp32_to_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kAny))})
    .Finalize();
