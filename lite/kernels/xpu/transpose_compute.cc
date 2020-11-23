// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/transpose_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void TransposeCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x = param.x;
  auto& axis = param.axis;
  int ndims = axis.size();
  const auto x_dims = x->dims();
  std::vector<int> x_shape_host(ndims, 0);

  for (int i = 0; i < ndims; ++i) {
    x_shape_host[i] = x_dims[i];
  }
  int r =
      xdnn::transpose<float>(ctx.GetRawContext(),
                             x->data<float>(),
                             param.output->mutable_data<float>(TARGET(kXPU)),
                             x_shape_host,
                             axis);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(transpose,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::TransposeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(transpose2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::TransposeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
