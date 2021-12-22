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

#include "lite/kernels/xpu/expand_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
void ExpandCompute<T>::Run() {
  auto& param = this->template Param<operators::ExpandParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  const auto* x = param.X;
  auto* out = param.Out;
  std::vector<int64_t> x_shape = x->dims().Vectorize();
  std::vector<int64_t> out_shape = out->dims().Vectorize();
  std::vector<int> x_dims(x_shape.begin(), x_shape.end());
  std::vector<int> out_dims(out_shape.begin(), out_shape.end());
  x_dims.insert(x_dims.begin(), out_dims.size() - x_dims.size(), 1);

  int r = xdnn::broadcast<T>(ctx.GetRawContext(),
                             x->template data<T>(),
                             out->template mutable_data<T>(TARGET(kXPU)),
                             x_dims,
                             out_dims);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using expand_xpu_float = paddle::lite::kernels::xpu::ExpandCompute<float>;
REGISTER_LITE_KERNEL(expand, kXPU, kFloat, kAny, expand_xpu_float, def_float)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("ExpandTimes",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using expand_xpu_int = paddle::lite::kernels::xpu::ExpandCompute<int>;
REGISTER_LITE_KERNEL(expand, kXPU, kFloat, kAny, expand_xpu_int, def_int)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("ExpandTimes",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();
