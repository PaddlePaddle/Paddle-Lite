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

#include "lite/kernels/xpu/stack_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void StackCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  int n = param.X.size();
  x_ptr_guard_ = TargetWrapperXPU::MallocScratchPad(
      n * 8 /* sizeof(__global__ float*) */, false /* use_l3 */);
  x_ptr_cpu_.reserve(n);
}

void StackCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int n = param.X.size();
  auto x_dims = param.X[0]->dims();
  int axis = param.axis;
  // XXX(miaotianxiang): +1?
  if (axis < 0) axis += (x_dims.size() + 1);
  auto matrix = x_dims.Flatten2D(axis);
  int height = matrix[0];
  int width = matrix[1];

  for (int i = 0; i < n; ++i) {
    x_ptr_cpu_[i] = param.X[i]->data<float>();
  }
  XPU_CALL(xpu_memcpy(
      x_ptr_guard_->addr_, &x_ptr_cpu_[0], n * 8, XPU_HOST_TO_DEVICE));

  int r = xdnn::stack_forward(
      ctx.GetRawContext(), /* context */
      height,              /* height */
      width,               /* width */
      n,                   /* n */
      x_ptr_guard_->addr_, /* x_ptr */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* out */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    stack, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::StackCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
