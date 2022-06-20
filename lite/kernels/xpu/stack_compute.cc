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

void StackCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int n = param.X.size();
  auto x_dims = param.X[0]->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis += (x_dims.size() + 1);
  }
  std::vector<int> x_shape;
  auto y_dim = param.Out->dims();
  for (int i = 0; i < y_dim.size(); i++) {
    x_shape.push_back(y_dim[i]);
  }
  x_shape[axis] = 1;
  std::vector<std::vector<int>> xdims_list(n, x_shape);

  std::vector<const float*> x_list(n, nullptr);
  for (int i = 0; i < n; ++i) {
    x_list[i] = param.X[i]->data<float>();
  }
  int r = xdnn::concat<float>(ctx.GetRawContext(),
                              x_list,
                              param.Out->mutable_data<float>(TARGET(kXPU)),
                              xdims_list,
                              axis);
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
