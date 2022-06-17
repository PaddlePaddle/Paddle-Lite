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

#include "lite/kernels/xpu/fill_any_like_compute.h"
#include <iostream>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FillAnyLikeCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int write_size = param.X->numel();

  int dtype = param.dtype;
  if (dtype == -1) {
    switch (param.X->precision()) {
      case PRECISION(kFloat):
        dtype = static_cast<int32_t>(lite::core::FluidType::FP32);
        break;
      case PRECISION(kInt32):
        dtype = static_cast<int32_t>(lite::core::FluidType::INT32);
        break;
      case PRECISION(kInt8):
        dtype = static_cast<int32_t>(lite::core::FluidType::INT8);
        break;
      case PRECISION(kInt64):
        dtype = static_cast<int32_t>(lite::core::FluidType::INT64);
        break;
      default:
        LOG(FATAL) << "not supported x dtype: "
                   << lite_api::PrecisionToStr(param.X->precision());
        break;
    }
  }

  int r = 0;
  switch (dtype) {
    case 1: {
      auto data = param.Out->mutable_data<int16_t>(TARGET(kXPU));
      r = xdnn::constant<int16_t>(ctx.GetRawContext(),
                                  data,
                                  write_size,
                                  static_cast<int16_t>(param.value));
      break;
    }
    case 2: {
      auto data = param.Out->mutable_data<int32_t>(TARGET(kXPU));
      r = xdnn::constant<int32_t>(ctx.GetRawContext(),
                                  data,
                                  write_size,
                                  static_cast<int32_t>(param.value));
      break;
    }
    case 3: {
      auto data = param.Out->mutable_data<int64_t>(TARGET(kXPU));
      r = xdnn::constant<int64_t>(ctx.GetRawContext(),
                                  data,
                                  write_size,
                                  static_cast<int64_t>(param.value));
      break;
    }
    case 5: {
      auto data = param.Out->mutable_data<float>(TARGET(kXPU));
      r = xdnn::constant<float>(ctx.GetRawContext(),
                                data,
                                write_size,
                                static_cast<float>(param.value));
      break;
    }
    default: {
      LOG(FATAL) << "Attribute dtype in fill_any_like op "
                    "must be 1[int16] or 3[int64] or 2[int32] or 5[fp32] "
                    "for xpu: "
                 << param.dtype;
      break;
    }
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fill_any_like,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FillAnyLikeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
