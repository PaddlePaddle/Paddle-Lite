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

#include "lite/kernels/xpu/__xpu__greater_than_filter_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GreaterThanFilterCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto input_dim = param.input->dims();
  CHECK_EQ(input_dim.size(), 2);
  int tensor_len = input_dim[0];

  int r = xdnn::greater_filter_fusion<float>(
      ctx.GetRawContext(),
      param.input->data<float>(),
      param.output->mutable_data<float>(TARGET(kXPU)),
      param.scale,
      tensor_len);
  CHECK_EQ(r, 0) << "call GreaterThanFilterCompute failed";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__greater_than_filter,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GreaterThanFilterCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
