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

#include "lite/kernels/xpu/pool_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void Pool2DCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.x->dims();
  CHECK_EQ(x_dims.size(), 4);
  auto& o_dims = param.output->dims();
  CHECK_EQ(param.ksize.size(), 2);
  if (param.global_pooling) {
    param.ksize[0] = x_dims[2];
    param.ksize[1] = x_dims[3];
  }
  CHECK_EQ(param.strides.size(), 2);
  CHECK_EQ(param.paddings->size(), 4);
  auto& paddings = *param.paddings;
  auto type = xdnn::MAX_WITHOUT_INDEX;
  if (param.pooling_type == "avg") {
    if (paddings[0] == 0 && paddings[1] == 0 && paddings[2] == 0 &&
        paddings[3] == 0) {
      type = xdnn::AVG_WITHOUT_PAD;
    } else {
      type = xdnn::AVG_WITH_PAD;
    }
  }

  int r = xdnn::pooling_forward<float, float>(
      ctx.GetRawContext(),                             /* context */
      param.x->data<float>(),                          /* x */
      param.output->mutable_data<float>(TARGET(kXPU)), /* y */
      nullptr,                                         /* y_index */
      type,                                            /* type */
      x_dims[0] * x_dims[1],                           /* c */
      x_dims[2],                                       /* in_h */
      x_dims[3],                                       /* in_w */
      paddings[0],                                     /* pad_left */
      paddings[1],                                     /* pad_right */
      paddings[2],                                     /* pad_up */
      paddings[3],                                     /* pad_down */
      param.ksize[0],                                  /* win_h */
      param.ksize[1],                                  /* win_w */
      param.strides[0],                                /* stride_h */
      param.strides[1],                                /* stride_w */
      o_dims[2],                                       /* out_h */
      o_dims[3] /* out_w */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pool2d, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::Pool2DCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
