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

template <typename T, PrecisionType PType>
void RangeCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::RangeParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  T start = (param.Start->template data<T>()[0]);
  T step = (param.Step->template data<T>()[0]);
  int64_t len = param.Out->numel();
  T* out_data = param.Out->template mutable_data<T>(TARGET(kXPU));
  int r = xdnn::range<T>(
      ctx.GetRawContext(), out_data, start, step, static_cast<int>(len));
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
    paddle::lite::kernels::xpu::RangeCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kXPU, kFloat, kAny, range_int64, range_int64)
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
    paddle::lite::kernels::xpu::RangeCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kXPU, kFloat, kAny, range_int32, range_int32)
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