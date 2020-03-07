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

#include "lite/kernels/arm/negative_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void NegativeCompute::PrepareForRun() {
  LOG(INFO) << "into negative kernels prepare run";
}

void NegativeCompute::Run() {
  LOG(INFO) << "into kernel compute run";
  auto& param = Param<param_t>();
  const auto* x_data = param.X->data<float>();
  auto* o_data = param.Out->mutable_data<float>();
  int num = param.X->dims().production();
  LOG(INFO) << "into negative fun";
  lite::arm::math::negative_func<float>(x_data, o_data, num);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(negative,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NegativeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
