// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/is_empty_compute.h"

#include <iostream>

#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void IsEmptyCompute::Run() {
  auto& param = this->template Param<operators::IsEmptyParam>();
  const size_t count = param.X->numel();
  auto out = param.Out->mutable_data<bool>(TARGET(kXPU));
  auto& ctx = this->ctx_->template As<XPUContext>();
  int ret = xdnn::constant<bool>(ctx.GetRawContext(), out, 1, count == 0);
  CHECK_EQ(ret, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    is_empty, kXPU, kAny, kAny, paddle::lite::kernels::xpu::IsEmptyCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();
