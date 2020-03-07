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

#include "lite/kernels/arm/fill_constant_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void FillConstantCompute::Run() {
  auto& param = *param_.get_mutable<param_t>();
  auto& context = ctx_->As<ARMContext>();

  if (param.dtype == static_cast<int32_t>(lite::core::FluidType::FP32)) {
    auto data = param.out->template mutable_data<float>();
    for (int i = 0; i < param.out->numel(); i++) {
      data[i] = param.value;
    }
  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT32)) {
    auto data = param.out->template mutable_data<int32_t>();
    for (int i = 0; i < param.out->numel(); i++) {
      data[i] = param.value;
    }
  } else if (param.dtype == static_cast<int32_t>(lite::core::FluidType::INT8)) {
    auto data = param.out->template mutable_data<int8_t>();
    for (int i = 0; i < param.out->numel(); i++) {
      data[i] = param.value;
    }
  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT64)) {
    auto data = param.out->template mutable_data<int64_t>();
    for (int i = 0; i < param.out->numel(); i++) {
      data[i] = param.value;
    }
  } else {
    LOG(FATAL) << "not supported dtype " << param.dtype;
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(fill_constant,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::FillConstantCompute,
                     def)
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
