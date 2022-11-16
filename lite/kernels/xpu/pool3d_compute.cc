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

#include <algorithm>
#include <vector>

#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/xpu/pool3d_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
template <typename InType, PrecisionType PType>
void Pool3DCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  CHECK_EQ(param.strides.size(), 3UL);
  CHECK_EQ(param.paddings->size(), 6UL);
  std::vector<int> paddings{(*param.paddings)[0],
                            (*param.paddings)[1],
                            (*param.paddings)[2],
                            (*param.paddings)[3],
                            (*param.paddings)[4],
                            (*param.paddings)[5]};
  if (param.ceil_mode) {
    paddings[1] += param.strides[0] - 1;
    paddings[3] += param.strides[1] - 1;
    paddings[5] += param.strides[2] - 1;
  }

  auto ksize = param.ksize;
  CHECK_EQ(ksize.size(), 3UL);
  auto& x_dims = param.x->dims();
  CHECK_EQ(x_dims.size(), 5);
  ksize[0] = (std::min)(
      ksize[0], static_cast<int>(x_dims[2]) + paddings[0] + paddings[1]);
  ksize[1] = (std::min)(
      ksize[1], static_cast<int>(x_dims[3]) + paddings[2] + paddings[3]);
  ksize[1] = (std::min)(
      ksize[2], static_cast<int>(x_dims[4]) + paddings[4] + paddings[5]);

  if (param.global_pooling) {
    ksize[0] = x_dims[2];
    ksize[1] = x_dims[3];
    ksize[2] = x_dims[4];
  }

  if (param.adaptive) {
    if (param.pooling_type == "avg") {
      int r = xdnn::adaptive_avg_pool3d(
          ctx.GetRawContext(),
          param.x->template data<InType>(),
          param.output->template mutable_data<InType>(TARGET(kXPU)),
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          x_dims[4],
          ksize[0],
          ksize[1],
          ksize[2],
          true);
      CHECK_EQ(r, 0);
    } else {
      int r = xdnn::adaptive_max_pool3d(
          ctx.GetRawContext(),
          param.x->template data<InType>(),
          param.output->template mutable_data<InType>(TARGET(kXPU)),
          nullptr,
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          x_dims[4],
          ksize[0],
          ksize[1],
          ksize[2],
          true);
      CHECK_EQ(r, 0);
    }
  } else {
    if (param.pooling_type == "avg") {
      int r = xdnn::avg_pool3d<InType>(
          ctx.GetRawContext(),
          param.x->template data<InType>(),
          param.output->template mutable_data<InType>(TARGET(kXPU)),
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          x_dims[4],
          ksize,
          param.strides,
          paddings,
          !param.exclusive,
          true);
      CHECK_EQ(r, 0);
    } else {
      if (param.pad_zero == true) {
        int r = xdnn::max_pool3d<InType>(
            ctx.GetRawContext(),
            param.x->template data<InType>(),
            param.output->template mutable_data<InType>(TARGET(kXPU)),
            nullptr,
            x_dims[0],
            x_dims[1],
            x_dims[2],
            x_dims[3],
            x_dims[4],
            ksize,
            param.strides,
            paddings,
            true,
            nullptr,
            nullptr);
        CHECK_EQ(r, 0);
      } else {
        int r = xdnn::max_pool3d<InType>(
            ctx.GetRawContext(),
            param.x->template data<InType>(),
            param.output->template mutable_data<InType>(TARGET(kXPU)),
            nullptr,
            x_dims[0],
            x_dims[1],
            x_dims[2],
            x_dims[3],
            x_dims[4],
            ksize,
            param.strides,
            paddings,
            true);
        CHECK_EQ(r, 0);
      }
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using pool3d_fp32 =
    paddle::lite::kernels::xpu::Pool3DCompute<float, PRECISION(kFloat)>;
using pool3d_fp16 =
    paddle::lite::kernels::xpu::Pool3DCompute<float16, PRECISION(kFP16)>;

using max_pool3d_with_index_fp32 =
    paddle::lite::kernels::xpu::Pool3DCompute<float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(pool3d, kXPU, kFloat, kNCHW, pool3d_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    pool3d, kXPU, kFP16, kNCHW, pool3d_fp16, DISABLE_XPU1_pool3d_FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    max_pool3d_with_index, kXPU, kFloat, kNCHW, max_pool3d_with_index_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
