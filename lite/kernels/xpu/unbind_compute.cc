// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/unbind_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void UnbindCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.x;
  auto& axis = param.axis;

  auto output = param.output;

  std::vector<T*> y_ptrs;
  for (size_t j = 0; j < output.size(); ++j) {
    y_ptrs.emplace_back(output[j]->template mutable_data<T>(TARGET(kXPU)));
  }
  auto x_shape = x->dims().Vectorize();
  int r = xdnn::unbind(
      ctx.GetRawContext(), x->template data<T>(), y_ptrs, x_shape, axis);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using unbind_fp32 =
    paddle::lite::kernels::xpu::UnbindCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unbind, kXPU, kFloat, kNCHW, unbind_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

using unbind_int64 =
    paddle::lite::kernels::xpu::UnbindCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unbind, kXPU, kFloat, kNCHW, unbind_int64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
