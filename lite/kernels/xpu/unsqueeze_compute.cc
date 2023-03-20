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

#include "lite/kernels/xpu/unsqueeze_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <PrecisionType PType>
void UnsqueezeCompute<PType>::Run() {
  auto& param = this->template Param<operators::UnsqueezeParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.X;
  auto output = param.Out;
  auto output_dims = output->dims();
  if (param.inplace) {
    output->ShareDataWith(*x);
  } else {
    output->set_precision(x->precision());
    output->template mutable_data(TARGET(kXPU), x->memory_size());
    int r = xdnn::copy<int8_t>(ctx.GetRawContext(),
                               x->template data<int8_t>(),
                               static_cast<int8_t*>(output->raw_data()),
                               x->memory_size());
    CHECK_EQ(r, 0);
  }
  output->Resize(output_dims);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    unsqueeze,
    kXPU,
    kAny,
    kAny,
    paddle::lite::kernels::xpu::UnsqueezeCompute<PRECISION(kAny)>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindInput("AxesTensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("AxesTensorList",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();

REGISTER_LITE_KERNEL(
    unsqueeze2,
    kXPU,
    kAny,
    kAny,
    paddle::lite::kernels::xpu::UnsqueezeCompute<PRECISION(kAny)>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindInput("AxesTensor",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("AxesTensorList",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();
