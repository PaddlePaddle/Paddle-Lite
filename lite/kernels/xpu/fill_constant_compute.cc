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

#include "lite/kernels/xpu/fill_constant_compute.h"

#include <iostream>

#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
int FillConstantCompute::FillConstData() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int write_size = param.out->numel();

  T value = static_cast<T>(param.value);
  if (param.value_tensor) {
    value = param.value_tensor->template mutable_data<T>()[0];
  }
  auto data = param.out->template mutable_data<T>(TARGET(kXPU));
  return xdnn::constant<T>(ctx.GetRawContext(), data, write_size, value);
}

void FillConstantCompute::Run() {
  auto& param = this->template Param<param_t>();
  int r = 0;
  switch (param.dtype) {
    case 0: {
      r = FillConstData<bool>();
      break;
    }
    case 1: {
      r = FillConstData<int16_t>();
      break;
    }
    case 2: {
      r = FillConstData<int>();
      break;
    }
    case 3: {
      r = FillConstData<int64_t>();
      break;
    }
    case 5: {
      r = FillConstData<float>();
      break;
    }
    default: {
      LOG(FATAL) << "Attribute dtype in fill_constant op "
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

REGISTER_LITE_KERNEL(fill_constant,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::FillConstantCompute,
                     def)
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("ValueTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindPaddleOpVersion("fill_constant", 2)
    .Finalize();
