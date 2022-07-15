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

#include "lite/kernels/xpu/pad_constant_like_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

union TypeUnion {
  float fp32;
  int32_t int32;
};

void PadConstantLikeCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x_dims = param.x->dims();
  auto y_dims = param.y->dims();
  float* out = param.output->mutable_data<float>(TARGET(kXPU));

  // x and y's dims should be same except the first dim
  if (x_dims.size() == 2 && x_dims[1] == y_dims[1] && param.pad_value <= 1e-5 &&
      param.pad_value >= -1e-5) {
    int r = xdnn::constant(
        ctx.GetRawContext(), out, param.x->numel(), param.pad_value);
    CHECK_EQ(r, 0);

    r = xdnn::add<float>(ctx.GetRawContext(),
                         param.y->data<float>(),
                         out,
                         out,
                         param.y->numel());

    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "xpu unsupport shape or value, x_dims " << x_dims
               << ", param.pad_value: " << param.pad_value;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pad_constant_like,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::PadConstantLikeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
