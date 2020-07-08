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

#include "lite/kernels/cuda/feed_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType Ptype>
void FeedCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  VLOG(4) << "feed_list.size: " << param.feed_list->size();
  const lite::Tensor& feed_item = (*param.feed_list)[param.col];

  int num = static_cast<int>(feed_item.numel());
  auto input = feed_item.data<T>();
  param.out->Resize(feed_item.dims());
  auto output = param.out->template mutable_data<T>(TARGET(kCUDA));
  VLOG(4) << "col: " << param.col << " num:" << num;

  TargetW::MemcpyAsync(
      output, input, num * sizeof(T), IoDirection::HtoD, stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::cuda::FeedCompute<float, PRECISION(kFloat)>
    FeedFp32;

typedef paddle::lite::kernels::cuda::FeedCompute<int64_t, PRECISION(kInt64)>
    FeedInt64;

typedef paddle::lite::kernels::cuda::FeedCompute<int32_t, PRECISION(kInt32)>
    FeedInt32;

REGISTER_LITE_KERNEL(feed, kCUDA, kFloat, kNCHW, FeedFp32, nchw)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(feed, kCUDA, kFloat, kNHWC, FeedFp32, nhwc)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(feed, kCUDA, kInt64, kNCHW, FeedInt64, nchw)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(feed, kCUDA, kInt64, kNHWC, FeedInt64, nhwc)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(feed, kCUDA, kInt32, kNCHW, FeedInt32, nchw)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(feed, kCUDA, kInt32, kNHWC, FeedInt32, nhwc)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
