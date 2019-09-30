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
#include "lite/backends/cuda/math/transpose.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename Dtype>
void NCHWToNHWCCompute<Dtype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto input = param.x->template data<Dtype>();
  auto input_dim = param.x->dims();
  CHECK(input_dim.size() == 4)
      << "NCHW to NHWC should guarantee that the input dims should be 4";
  auto output = param.y->template mutable_data<Dtype>(TARGET(kCUDA));

  int n = input_dim[0];
  int c = input_dim[1];
  int h = input_dim[2];
  int w = input_dim[3];

  lite::cuda::math::NCHW2NHWC<Dtype>(n, c, h * w, input, output, &ctx);
}

template <typename Dtype>
void NHWCToNCHWCompute<Dtype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();

  auto input = param.x->template data<Dtype>();
  auto output = param.y->template mutable_data<Dtype>(TARGET(kCUDA));

  auto input_dim = param.x->dims();
  CHECK(input_dim.size() == 4)
      << "NHWC to NCHW should guarantee that the input dims should be 4";

  int n = input_dim[0];
  int h = input_dim[1];
  int w = input_dim[2];
  int c = input_dim[3];
  lite::cuda::math::NHWC2NCHW<Dtype>(n, c, h * w, input, output, &ctx);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(layout,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::NCHWToNHWCCompute<float>,
                     nchw2nhwc)
    .BindInput("X",
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
                     kNHWC,
                     paddle::lite::kernels::cuda::NHWCToNCHWCompute<float>,
                     nhwc2nchw)
    .BindInput("X",
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
                     paddle::lite::kernels::cuda::NCHWToNHWCCompute<int8_t>,
                     nchw2nhwc)
    .BindInput("X",
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
                     kNHWC,
                     paddle::lite::kernels::cuda::NHWCToNCHWCompute<int8_t>,
                     nhwc2nchw)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
