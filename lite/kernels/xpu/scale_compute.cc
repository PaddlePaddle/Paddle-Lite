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

#include "lite/kernels/xpu/scale_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void ScaleCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& x_dims = param.x->dims();
  if (std::fabs(param.scale - 1.0f) < 1e-7 && std::fabs(param.bias) < 1e-7) {
    auto x = param.x;
    param.output->ShareDataWith(*x);
    param.output->Resize(param.output->dims());
  } else {
    int r = xdnn::scale<T>(
        ctx.GetRawContext(),
        param.x->template data<T>(),                          /* x */
        param.output->template mutable_data<T>(TARGET(kXPU)), /* y */
        x_dims.production(),                                  /* len */
        param.bias_after_scale, /* bias_after_scale */
        param.scale,            /* alpha */
        param.bias);            /* beta */
    CHECK_EQ(r, 0);
  }
  if (!param.x->lod().empty()) {
    param.output->set_lod(param.x->lod());
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using XPUScale_FP32 =
    paddle::lite::kernels::xpu::ScaleCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(scale, kXPU, kFloat, kNCHW, XPUScale_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using XPUScale_FP16 =
    paddle::lite::kernels::xpu::ScaleCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(scale, kXPU, kFP16, kNCHW, XPUScale_FP16, fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using XPUScale_Int32 =
    paddle::lite::kernels::xpu::ScaleCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(scale, kXPU, kFloat, kNCHW, XPUScale_Int32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

using XPUScale_Int64 =
    paddle::lite::kernels::xpu::ScaleCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(scale, kXPU, kFloat, kNCHW, XPUScale_Int64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
