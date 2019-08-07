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

#include "lite/kernels/arm/cast_compute.h"
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void CastCompute::PrepareForRun() {}

void CastCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::CastParam>();

  auto input_dims = param.X->dims();

  if (param.in_dtype == param.out_dtype && param.in_dtype == 2 ||
      param.in_dtype == 0) {
    const auto* x_data = param.X->data<float>();
    auto* o_data = param.Out->mutable_data<float>();
    memcpy(o_data, x_data, sizeof(float) * param.X->numel());
  } else {
    LOG(FATAL) << "other has not been implemented";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cast, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::CastCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
