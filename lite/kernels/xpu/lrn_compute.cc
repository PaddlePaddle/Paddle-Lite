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

#include "lite/kernels/xpu/lrn_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void LrnCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x_dims = param.X->dims();
  int batch = x_dims[0];
  int channel = x_dims[1];
  int h = x_dims[2];
  int w = x_dims[3];
  int n = param.n;
  float alpha = param.alpha;
  float beta = param.beta;
  float k = param.k;
  if (param.norm_region == "AcrossChannels") {
    int r = xdnn::lrn(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      batch,
                      channel,
                      h,
                      w,
                      n,
                      k,
                      alpha,
                      beta);
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "Unsupport Norm Region Type: " << param.norm_region;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    lrn, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::LrnCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("MidOut", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
