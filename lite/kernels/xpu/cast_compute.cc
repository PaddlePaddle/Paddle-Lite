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
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename in_type>
int XpuCastCompute(xdnn::Context* ctx, const operators::CastParam& param) {
  auto* in_data = param.X->data<in_type>();
  int numel = param.X->numel();
  int ret = -1;
  int out_type = param.out_dtype;
  if (out_type == 2) {
    auto* out_data = param.Out->mutable_data<int>(TARGET(kXPU));
    ret = xdnn::cast_v2<in_type, int>(ctx, in_data, out_data, numel);
  } else if (out_type == 3) {
    auto* out_data = param.Out->mutable_data<int64_t>(TARGET(kXPU));
    ret = xdnn::cast_v2<in_type, int64_t>(ctx, in_data, out_data, numel);
  } else if (out_type == 5) {
    auto* out_data = param.Out->mutable_data<float>(TARGET(kXPU));
    ret = xdnn::cast_v2<in_type, float>(ctx, in_data, out_data, numel);
  }
  return ret;
}

void CastCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  if (param.X->numel() <= 0) {
    return;
  }
  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  auto precision = param.X->precision();
  switch (precision) {
    case PRECISION(kInt32): {
      int r = XpuCastCompute<int>(ctx.GetRawContext(), param);
      CHECK_EQ(r, 0);
      break;
    }
    case PRECISION(kInt64): {
      int r = XpuCastCompute<int64_t>(ctx.GetRawContext(), param);
      CHECK_EQ(r, 0);
      break;
    }
    case PRECISION(kFloat): {
      int r = XpuCastCompute<float>(ctx.GetRawContext(), param);
      CHECK_EQ(r, 0);
      break;
    }
    default: {
      LOG(FATAL) << "unsupported data precision: "
                 << lite_api::PrecisionToStr(precision);
      break;
    }
  }
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
