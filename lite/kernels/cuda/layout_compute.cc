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

#include "lite/kernels/cuda/layout_compute.h"
#include <vector>
#include "lite/backends/cuda/math/transpose.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

inline DDim trim_singular_dims(const DDim& dims) {
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }
  std::vector<int64_t> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (size_t i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return DDim();
  }
  return DDim(trim_dims);
}

#define NCHWTONHWC(type)                                                  \
  auto& param = this->template Param<param_t>();                          \
  auto& ctx = this->ctx_->template As<CUDAContext>();                     \
  auto stream = ctx.exec_stream();                                        \
  auto input = param.x->template data<type>();                            \
  auto input_dim = param.x->dims();                                       \
  DDim input_trim_dim = trim_singular_dims(input_dim);                    \
  if (input_trim_dim.size() == 1) {                                       \
    param.y->CopyDataFrom(*param.x);                                      \
    return;                                                               \
  }                                                                       \
  CHECK(input_dim.size() == 4)                                            \
      << "NCHW to NHWC should guarantee that the input dims should be 4"; \
  int n = input_dim[0];                                                   \
  int c = input_dim[1];                                                   \
  int h = input_dim[2];                                                   \
  int w = input_dim[3];                                                   \
  param.y->Resize({n, h, w, c});                                          \
  auto output = param.y->template mutable_data<type>(TARGET(kCUDA));      \
  trans.NCHW2NHWC(n, c, h* w, input, output, &stream);

#define NHWCTONCHW(type)                                                  \
  auto& param = this->template Param<param_t>();                          \
  auto& ctx = this->ctx_->template As<CUDAContext>();                     \
  auto stream = ctx.exec_stream();                                        \
  auto input = param.x->template data<type>();                            \
  auto input_dim = param.x->dims();                                       \
  DDim input_trim_dim = trim_singular_dims(input_dim);                    \
  if (input_trim_dim.size() == 1) {                                       \
    param.y->CopyDataFrom(*param.x);                                      \
    return;                                                               \
  }                                                                       \
  CHECK(input_dim.size() == 4)                                            \
      << "NHWC to NCHW should guarantee that the input dims should be 4"; \
  int n = input_dim[0];                                                   \
  int h = input_dim[1];                                                   \
  int w = input_dim[2];                                                   \
  int c = input_dim[3];                                                   \
  param.y->Resize({n, c, h, w});                                          \
  auto output = param.y->template mutable_data<type>(TARGET(kCUDA));      \
  trans.NHWC2NCHW(n, c, h* w, input, output, &stream);

void NCHWToNHWCCompute::Run() { NCHWTONHWC(float) }

void NCHWToNHWCComputeInt8::Run() { NCHWTONHWC(int8_t) }

void NHWCToNCHWCompute::Run() { NHWCTONCHW(float) }

void NHWCToNCHWComputeInt8::Run() { NHWCTONCHW(int8_t) }

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(layout,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::NCHWToNHWCCompute,
                     nchw2nhwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(layout,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::NHWCToNCHWCompute,
                     nhwc2nchw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(layout,
                     kCUDA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::cuda::NCHWToNHWCComputeInt8,
                     int8_nchw2nhwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(layout,
                     kCUDA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::cuda::NHWCToNCHWComputeInt8,
                     int8_nhwc2nchw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(layout_once,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::NCHWToNHWCCompute,
                     nchw2nhwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(layout_once,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::NHWCToNCHWCompute,
                     nhwc2nchw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(layout_once,
                     kCUDA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::cuda::NCHWToNHWCComputeInt8,
                     int8_nchw2nhwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(layout_once,
                     kCUDA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::cuda::NHWCToNCHWComputeInt8,
                     int8_nhwc2nchw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
