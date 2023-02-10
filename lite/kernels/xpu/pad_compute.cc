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

#include "lite/kernels/xpu/pad_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
void PadCompute<T>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& pads = param.paddings;
  auto* x = param.X;
  for (int i = 0; i < x->dims().size(); i++) {
    pad_left_.push_back(pads[i * 2]);
    pad_right_.push_back(pads[i * 2 + 1]);
  }
}

template <typename T>
void PadCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  T value = static_cast<T>(param.pad_value);

  auto* x = param.X;
  auto in_dims = x->dims();
  auto* in_data = x->template data<T>();
  auto* out = param.Out;
  T* out_data = out->template mutable_data<T>(TARGET(kXPU));

  int r = xdnn::pad<T>(ctx.GetRawContext(),
                       in_data,
                       out_data,
                       x->dims().data(),
                       pad_left_,
                       pad_right_,
                       value);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pad,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::PadCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
