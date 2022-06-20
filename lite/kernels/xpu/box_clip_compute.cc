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

#include "lite/kernels/xpu/box_clip_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void BoxClipCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const auto* input = param.Input;
  const auto* im_info = param.ImInfo;
  float h = im_info->data<float>()[0] / im_info->data<float>()[2];
  float w = im_info->data<float>()[1] / im_info->data<float>()[2];
  int r = xdnn::clip_box_to_image<float>(
      ctx.GetRawContext(),
      input->data<float>(),
      param.Output->mutable_data<float>(TARGET(kXPU)),
      static_cast<int>(input->numel() / 4),
      static_cast<int>(h),
      static_cast<int>(w));
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_clip,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::BoxClipCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ImInfo", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
