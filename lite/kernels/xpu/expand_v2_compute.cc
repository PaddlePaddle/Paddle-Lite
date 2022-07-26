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

#include "lite/kernels/xpu/expand_v2_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void ExpandV2Compute<T, PType>::Run() {
  auto& param = this->template Param<operators::ExpandV2Param>();
  auto& ctx = this->ctx_->template As<XPUContext>();
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

using expand_v2_xpu_float =
    paddle::lite::kernels::xpu::ExpandV2Compute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand_v2, kXPU, kFloat, kAny, expand_v2_xpu_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_shapes_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using expand_v2_xpu_fp16 =
    paddle::lite::kernels::xpu::ExpandV2Compute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(expand_v2, kXPU, kFP16, kAny, expand_v2_xpu_fp16, fp16)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_shapes_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kAny))})
    .Finalize();
