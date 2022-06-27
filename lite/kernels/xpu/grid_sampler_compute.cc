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

#include "lite/kernels/xpu/grid_sampler_compute.h"
#include <string>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GridSamplerCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* input = param.x;
  auto* grid = param.grid;
  auto* output = param.out;
  const std::string padding_mode = param.padding_mode;
  const std::string mode = param.mode;
  const bool align_corners = param.align_corners;

  int n = input->dims()[0];
  int c = input->dims()[1];
  int xh = input->dims()[2];
  int xw = input->dims()[3];
  int yh = output->dims()[2];
  int yw = output->dims()[3];

  bool is_nearest = mode == "nearest" ? true : false;
  int pad_mode = 0;
  if (padding_mode == "border") {
    pad_mode = 1;
  } else if (padding_mode == "reflection") {
    pad_mode = 2;
  }

  int ret = xdnn::grid_sample<float>(ctx.GetRawContext(),
                                     input->data<float>(),
                                     grid->data<float>(),
                                     output->mutable_data<float>(TARGET(kXPU)),
                                     n,
                                     c,
                                     xh,
                                     xw,
                                     yh,
                                     yw,
                                     is_nearest,
                                     align_corners,
                                     pad_mode,
                                     true);
  CHECK_EQ(ret, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(grid_sampler,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GridSamplerCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Grid", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
