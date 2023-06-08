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

#include "lite/kernels/xpu/lookup_table_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename WType, typename IDType, PrecisionType PType>
void LookupTableCompute<WType, IDType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int ym = param.Ids->numel();
  int xm = param.W->dims()[0];
  int n = param.W->dims()[1];

  int r = xdnn::embedding<WType, IDType>(
      ctx.GetRawContext(), /* context */
      param.W->template data<WType>(),
      param.Ids->template data<IDType>(),
      param.Out->template mutable_data<WType>(TARGET(kXPU)), /* top */
      xm,
      n,
      ym,
      param.padding_idx);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using lookup_table_FP_INT64 =
    paddle::lite::kernels::xpu::LookupTableCompute<float,
                                                   int64_t,
                                                   PRECISION(kFloat)>;
using lookup_table_v2_FP_INT64 =
    paddle::lite::kernels::xpu::LookupTableCompute<float,
                                                   int64_t,
                                                   PRECISION(kFloat)>;
using lookup_table_v2_FP_INT32 =
    paddle::lite::kernels::xpu::LookupTableCompute<float,
                                                   int32_t,
                                                   PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    lookup_table, kXPU, kFloat, kNCHW, lookup_table_FP_INT64, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(lookup_table_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     lookup_table_v2_FP_INT64,
                     lookup_int64)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindPaddleOpVersion("lookup_table_v2", 1)
    .Finalize();

REGISTER_LITE_KERNEL(lookup_table_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     lookup_table_v2_FP_INT32,
                     lookup_int32)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindPaddleOpVersion("lookup_table_v2", 1)
    .Finalize();
