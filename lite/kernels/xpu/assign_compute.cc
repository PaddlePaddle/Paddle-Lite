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

#include "lite/kernels/xpu/assign_compute.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T, PrecisionType PType>
void AssignCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.X) << "only support input is tensor";
  if (param.X == param.Out || param.X->numel() == 0) {
    param.Out->set_target(TARGET(kXPU));
    return;
  }

  auto& ctx = this->ctx_->template As<XPUContext>();
  int r = xdnn::copy<T>(ctx.GetRawContext(),
                        param.X->template data<T>(),
                        param.Out->template mutable_data<T>(TARGET(kXPU)),
                        param.X->numel());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using assign_float =
    paddle::lite::kernels::xpu::AssignCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(assign, kXPU, kFloat, kNCHW, assign_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using assign_fp16 =
    paddle::lite::kernels::xpu::AssignCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(assign, kXPU, kFP16, kNCHW, assign_fp16, fp16)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kAny))})
    .Finalize();

using assign_int =
    paddle::lite::kernels::xpu::AssignCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(assign, kXPU, kFloat, kNCHW, assign_int, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using assign_int64 =
    paddle::lite::kernels::xpu::AssignCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(assign, kXPU, kFloat, kNCHW, assign_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();

using assign_int8 =
    paddle::lite::kernels::xpu::AssignCompute<int8_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(assign, kXPU, kFloat, kNCHW, assign_int8, bool)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();
