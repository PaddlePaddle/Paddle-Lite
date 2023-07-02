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

#include "lite/kernels/xpu/reshape_compute.h"
#include <algorithm>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <PrecisionType PType>
void ReshapeCompute<PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.x;
  auto output = param.output;
  auto output_dims = output->dims();
  if (output_dims.production() == 0) {
    output->set_target(TARGET(kXPU));
    return;
  }

  if (param.inplace) {
    auto output_lod = output->lod();
    output->ShareDataWith(*x);
    output->set_lod(output_lod);
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
    reshape2,
    kXPU,
    kAny,
    kAny,
    paddle::lite::kernels::xpu::ReshapeCompute<PRECISION(kAny)>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(
    reshape,
    kXPU,
    kAny,
    kAny,
    paddle::lite::kernels::xpu::ReshapeCompute<PRECISION(kAny)>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    flatten,
    kXPU,
    kAny,
    kAny,
    paddle::lite::kernels::xpu::ReshapeCompute<PRECISION(kAny)>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    flatten2,
    kXPU,
    kAny,
    kAny,
    paddle::lite::kernels::xpu::ReshapeCompute<PRECISION(kAny)>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
