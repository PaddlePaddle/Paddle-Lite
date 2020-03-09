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

#include "lite/kernels/arm/increment_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void IncrementCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::IncrementParam>();

  int total_num = param.X->dims().production();
  if (param.X->precision() == PRECISION(kFloat)) {
    const auto* x_data = param.X->data<float>();
    auto* o_data = param.Out->mutable_data<float>();
    lite::arm::math::increment(x_data, total_num, param.step, o_data, &ctx);
  } else if (param.X->precision() == PRECISION(kInt64)) {
    const auto* x_data = param.X->data<int64_t>();
    auto* o_data = param.Out->mutable_data<int64_t>();
    lite::arm::math::increment(x_data, total_num, param.step, o_data, &ctx);
  } else if (param.X->precision() == PRECISION(kInt32)) {
    const auto* x_data = param.X->data<int32_t>();
    auto* o_data = param.Out->mutable_data<int32_t>();
    lite::arm::math::increment(x_data, total_num, param.step, o_data, &ctx);
  } else {
    LOG(FATAL) << "unsupport input type "
               << PrecisionToStr(param.X->precision());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(increment,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::IncrementCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
