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

template <class T>
void ScaleCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& x_dims = param.x->dims();

  int r =
      xdnn::scale(ctx.GetRawContext(),
                  param.x->template data<T>(),                          /* x */
                  param.output->template mutable_data<T>(TARGET(kXPU)), /* y */
                  x_dims.production(),    /* len */
                  param.bias_after_scale, /* bias_after_scale */
                  param.scale,            /* alpha */
                  param.bias);            /* beta */
  CHECK_EQ(r, 0);
  if (!param.x->lod().empty()) {
    param.output->set_lod(param.x->lod());
  }
}

template <>
void ScaleCompute<int>::Run() {
  auto& param = this->template Param<param_t>();
  const int* x_data = param.x->template data<int>();
  int* out_data = param.output->template mutable_data<int>();
  int64_t size = param.x->numel();
  int scale = static_cast<int>(param.scale);
  int bias = static_cast<int>(param.bias);

  if (param.bias_after_scale) {
    for (int64_t i = 0; i < size; i++) {
      out_data[i] = x_data[i] * scale + bias;
    }
  } else {
    for (int64_t i = 0; i < size; i++) {
      out_data[i] = (x_data[i] + bias) * scale;
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(scale,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ScaleCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(scale,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ScaleCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
