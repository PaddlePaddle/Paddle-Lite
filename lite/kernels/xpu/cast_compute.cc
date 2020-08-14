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

#include "lite/kernels/xpu/cast_compute.h"
#include <typeinfo>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void CastCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* out = param.Out;
  int out_dtype = param.out_dtype;
  int in_dtype = param.in_dtype;
  int numel = param.X->numel();
  int r = 0;
  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;

  if (in_dtype == 5 && out_dtype == 5) {
    // float -> float
    auto* in_data = param.X->data<float>();
    auto* out_data = out->mutable_data<float>(TARGET(kXPU));
    r = xdnn::cast<float, float>(ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 2 && out_dtype == 2) {
    // int -> int
    auto* in_data = param.X->data<int>();
    auto* out_data = out->mutable_data<int>(TARGET(kXPU));
    r = xdnn::cast<int, int>(ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 3 && out_dtype == 3) {
    // int64 -> int64
    auto* in_data = param.X->data<int64_t>();
    auto* out_data = out->mutable_data<int64_t>(TARGET(kXPU));
    r = xdnn::cast<int64_t, int64_t>(
        ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 2 && out_dtype == 3) {
    // int -> int64
    auto* in_data = param.X->data<int>();
    auto* out_data = out->mutable_data<int64_t>(TARGET(kXPU));
    r = xdnn::cast<int, int64_t>(ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 2 && out_dtype == 5) {
    // int -> float
    auto* in_data = param.X->data<int>();
    auto* out_data = out->mutable_data<float>(TARGET(kXPU));
    r = xdnn::cast<int, float>(ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 3 && out_dtype == 5) {
    // int64_t -> float
    auto* in_data = param.X->data<int64_t>();
    auto* out_data = out->mutable_data<float>(TARGET(kXPU));
    r = xdnn::cast<int64_t, float>(
        ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 5 && out_dtype == 3) {
    // float -> int64_t
    auto* in_data = param.X->data<float>();
    auto* out_data = out->mutable_data<int64_t>(TARGET(kXPU));
    r = xdnn::cast<float, int64_t>(
        ctx.GetRawContext(), in_data, out_data, numel);
  } else if (in_dtype == 5 && out_dtype == 2) {
    // float -> int
    auto* in_data = param.X->data<float>();
    auto* out_data = out->mutable_data<int>(TARGET(kXPU));
    r = xdnn::cast<float, int>(ctx.GetRawContext(), in_data, out_data, numel);
  } else {
    CHECK(false);
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cast, kXPU, kAny, kNCHW, paddle::lite::kernels::xpu::CastCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
