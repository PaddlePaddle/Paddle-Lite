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

#include "lite/kernels/xpu/roll_compute.h"
#include <algorithm>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void RollCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  
  const auto* x = param.X;

  std::vector<int64_t> xshape=x->dims().Vectorize();

  auto* out = param.Out;
  std::vector<int64_t> axis = param.axis;
  std::vector<int64_t> shifts = param.shifts;

  int r = xdnn::roll<T>(ctx.GetRawContext(),
                        x->template data<T>(),
                        out->template mutable_data<T>(TARGET(kXPU)),
                        xshape,
                        shifts,
                        axis);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using rollFP32 =
    paddle::lite::kernels::xpu::RollCompute<float, PRECISION(kFloat)>;
using rollFP16 =
    paddle::lite::kernels::xpu::RollCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(roll, kXPU, kFloat, kAny, rollFP32, float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(roll, kXPU, kFP16, kAny, rollFP16, float16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();