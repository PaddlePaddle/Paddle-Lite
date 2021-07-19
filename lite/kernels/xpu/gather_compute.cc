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

#include "lite/kernels/xpu/gather_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GatherCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto x = param.X;
  auto index = param.Index;
  auto out = param.Out;
  if (out->numel() == 0) return;
  int axis = 0;
  if (param.Axis != nullptr) {
    CHECK(param.Axis->precision() == PRECISION(kInt32))
        << " xpu only support axis int32 type";
    auto* axis_data = param.Axis->data<int>();
    axis = axis_data[0];
  }
  CHECK_GE(axis, 0) << " xpu gather kernel not support axis < 0";
  std::vector<int> x_dims(x->dims().data().begin(), x->dims().data().end());

  int r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                   x->data<float>(),
                                   index->data<int>(),
                                   out->mutable_data<float>(TARGET(kXPU)),
                                   x_dims,
                                   index->numel(),
                                   axis);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::GatherCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
