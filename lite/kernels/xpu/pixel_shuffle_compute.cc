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

#include "lite/kernels/xpu/pixel_shuffle_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void PixelShuffleCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::PixelShuffleParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  const T* x_data = param.x->template data<T>();
  int upscale_factor = param.upscale_factor;
  auto in_dims = param.x->dims();

  T* output_data = param.output->template mutable_data<T>(TARGET(kXPU));

  int r = xdnn::pixel_shuffle<T>(ctx.GetRawContext(),
                                 x_data,
                                 output_data,
                                 static_cast<int>(in_dims[0]),
                                 static_cast<int>(in_dims[1]),
                                 static_cast<int>(in_dims[2]),
                                 static_cast<int>(in_dims[3]),
                                 upscale_factor,
                                 true);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using pixel_shuffle_float =
    paddle::lite::kernels::xpu::PixelShuffleCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    pixel_shuffle, kXPU, kFloat, kNCHW, pixel_shuffle_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

using pixel_shuffle_fp16 =
    paddle::lite::kernels::xpu::PixelShuffleCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    pixel_shuffle, kXPU, kFP16, kNCHW, pixel_shuffle_fp16, fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
