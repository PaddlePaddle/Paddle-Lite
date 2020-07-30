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

union TypeUnion {
  float fp32;
  int32_t int32;
};

void FillConstantCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  TypeUnion value;
  int write_size = param.out->numel();

  if (param.dtype == static_cast<int32_t>(lite::core::FluidType::FP32)) {
    auto data = param.out->template mutable_data<float>(TARGET(kXPU));
    value.fp32 = param.value;
    write_size = write_size * sizeof(float);
    int r = xdnn::memset(ctx.GetRawContext(), /* context */
                         reinterpret_cast<void*>(data),
                         value.int32,
                         write_size);
    CHECK_EQ(r, 0);

  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT32)) {
    auto data = param.out->template mutable_data<int32_t>(TARGET(kXPU));
    value.int32 = param.value;
    write_size = write_size * sizeof(int32_t);
    int r = xdnn::memset(ctx.GetRawContext(), /* context */
                         reinterpret_cast<void*>(data),
                         value.int32,
                         write_size);
    CHECK_EQ(r, 0);

  } else if (param.dtype == static_cast<int32_t>(lite::core::FluidType::INT8)) {
    auto data = param.out->template mutable_data<int8_t>(TARGET(kXPU));
    value.int32 = 0;
    for (int i = 0; i < 4; i++) {
      value.int32 += static_cast<int32_t>(param.value);
      value.int32 = value.int32 << 8;
    }
    int r = xdnn::memset(ctx.GetRawContext(), /* context */
                         reinterpret_cast<void*>(data),
                         value.int32,
                         write_size);
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "not supported dtype " << param.dtype;
  }
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
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
