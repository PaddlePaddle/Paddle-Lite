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
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void Pool2DCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  CHECK_EQ(param.strides.size(), 2UL);
  CHECK_EQ(param.paddings->size(), 4UL);
  std::vector<int> paddings{(*param.paddings)[0],
                            (*param.paddings)[1],
                            (*param.paddings)[2],
                            (*param.paddings)[3]};
  if (param.ceil_mode) {
    paddings[1] += param.strides[0] - 1;
    paddings[3] += param.strides[1] - 1;
  }

  auto ksize = param.ksize;
  CHECK_EQ(ksize.size(), 2UL);
  auto& x_dims = param.x->dims();
  CHECK_EQ(x_dims.size(), 4);
  ksize[0] = (std::min)(
      ksize[0], static_cast<int>(x_dims[2]) + paddings[0] + paddings[1]);
  ksize[1] = (std::min)(
      ksize[1], static_cast<int>(x_dims[3]) + paddings[2] + paddings[3]);
  if (param.global_pooling) {
    ksize[0] = x_dims[2];
    ksize[1] = x_dims[3];
  }

  if (param.adaptive) {
    if (param.pooling_type == "avg") {
      int r = xdnn::adaptive_avg_pool2d(
          ctx.GetRawContext(),
          param.x->data<float>(),
          param.output->mutable_data<float>(TARGET(kXPU)),
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          ksize[0],
          ksize[1],
          true);
      CHECK_EQ(r, 0);
    } else {
      int r = xdnn::adaptive_max_pool2d(
          ctx.GetRawContext(),
          param.x->data<float>(),
          param.output->mutable_data<float>(TARGET(kXPU)),
          nullptr,
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          ksize[0],
          ksize[1],
          true);
      CHECK_EQ(r, 0);
    }
  } else {
    if (param.pooling_type == "avg") {
      int r = xdnn::avg_pool2d<float>(
          ctx.GetRawContext(),
          param.x->data<float>(),
          param.output->mutable_data<float>(TARGET(kXPU)),
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          ksize,
          param.strides,
          paddings,
          !param.exclusive,
          true);
      CHECK_EQ(r, 0);
    } else {
      if (param.pad_zero == true) {
        int r = xdnn::max_pool2d<float>(
            ctx.GetRawContext(),
            param.x->data<float>(),
            param.output->mutable_data<float>(TARGET(kXPU)),
            nullptr,
            x_dims[0],
            x_dims[1],
            x_dims[2],
            x_dims[3],
            ksize,
            param.strides,
            paddings,
            true);
        CHECK_EQ(r, 0);
      } else {
        const float* xpu_x_padded = nullptr;
        std::vector<int> xpu_x_padded_dims{static_cast<int>(x_dims[0]),
                                           static_cast<int>(x_dims[1]),
                                           static_cast<int>(x_dims[2]),
                                           static_cast<int>(x_dims[3])};
        XPUScratchPadGuard xpu_x_padded_guard_;
        if (paddings[0] == 0 && paddings[1] == 0 && paddings[2] == 0 &&
            paddings[3] == 0) {
          xpu_x_padded = param.x->data<float>();
        } else {
          std::vector<int> pad_left{0, 0, paddings[0], paddings[2]};
          std::vector<int> pad_right{0, 0, paddings[1], paddings[3]};
          xpu_x_padded_dims[2] =
              xpu_x_padded_dims[2] + paddings[0] + paddings[1];
          xpu_x_padded_dims[3] =
              xpu_x_padded_dims[3] + paddings[2] + paddings[3];
          xpu_x_padded_guard_ = TargetWrapperXPU::MallocScratchPad(
              sizeof(float) * xpu_x_padded_dims[0] * xpu_x_padded_dims[1] *
              xpu_x_padded_dims[2] * xpu_x_padded_dims[3]);
          xpu_x_padded = reinterpret_cast<float*>(xpu_x_padded_guard_->addr_);
          int r = xdnn::pad<float>(ctx.GetRawContext(),
                                   param.x->data<float>(),
                                   const_cast<float*>(xpu_x_padded),
                                   {static_cast<int>(x_dims[0]),
                                    static_cast<int>(x_dims[1]),
                                    static_cast<int>(x_dims[2]),
                                    static_cast<int>(x_dims[3])},
                                   pad_left,
                                   pad_right,
                                   -9999999.0f);
          CHECK_EQ(r, 0);
        }
        int r = xdnn::max_pool2d<float>(
            ctx.GetRawContext(),
            xpu_x_padded,
            param.output->mutable_data<float>(TARGET(kXPU)),
            nullptr,
            xpu_x_padded_dims[0],
            xpu_x_padded_dims[1],
            xpu_x_padded_dims[2],
            xpu_x_padded_dims[3],
            ksize,
            param.strides,
            {0, 0, 0, 0},
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

REGISTER_LITE_KERNEL(
    pool2d, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::Pool2DCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(max_pool2d_with_index,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::Pool2DCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
