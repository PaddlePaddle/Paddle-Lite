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

void LookupTableCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int num = param.Ids->numel();
  int embed_dim = param.W->dims()[1];

  int r = xdnn::embedding<float, int64_t>(
      ctx.GetRawContext(),                          /* context */
      num,                                          /* num */
      param.Ids->data<int64_t>(),                   /* indices */
      embed_dim,                                    /* embed_dim */
      param.W->data<float>(),                       /* table */
      param.Out->mutable_data<float>(TARGET(kXPU)), /* top */
      param.padding_idx /* padding_idx */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lookup_table,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::LookupTableCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
