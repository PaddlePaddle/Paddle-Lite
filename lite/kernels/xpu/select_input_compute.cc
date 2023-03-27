// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/select_input_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SelectInputCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.X;

  auto output = param.Out;
  auto x_i = x[*param.Mask->data<int>()];
  output->mutable_data(TARGET(kXPU), x_i->memory_size());
  int r = xdnn::copy<int8_t>(ctx.GetRawContext(),
                             x_i->data<int8_t>(),
                             reinterpret_cast<int8_t*>(output->raw_data()),
                             x_i->memory_size());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(select_input,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::SelectInputCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindInput("Mask",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
