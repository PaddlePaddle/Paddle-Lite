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

#include "lite/kernels/xpu/range_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T, PrecisionType PType>
void RangeCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::RangeParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int r = xdnn::range<T>(ctx.GetRawContext(),
                         param.Out->template mutable_data<T>(TARGET(kXPU)),
                         param.Start->template data<T>()[0],
                         param.Step->template data<T>()[0],
                         param.Out->dims().production());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using range_float =
    paddle::lite::kernels::xpu::RangeCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kXPU, kFloat, kAny, range_float, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using range_int64 =
    paddle::lite::kernels::xpu::RangeCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(range, kXPU, kInt64, kAny, range_int64, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();

using range_int32 =
    paddle::lite::kernels::xpu::RangeCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(range, kXPU, kInt32, kAny, range_int32, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
// float kernel has higher score when picking kernel.
using range_int32_f =
    paddle::lite::kernels::xpu::RangeCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kXPU, kFloat, kAny, range_int32_f, int32)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

// float kernel has higher score when picking kernel.
using range_int64_f =
    paddle::lite::kernels::xpu::RangeCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kXPU, kFloat, kAny, range_int64_f, int64)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
