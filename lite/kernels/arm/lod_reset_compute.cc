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

#include "lite/kernels/arm/lod_reset_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
void LodResetCompute::PrepareForRun() {}

void LodResetCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::LodResetParam>();
  param.Out->CopyDataFrom(*param.X);
  auto lod = param.Out->mutable_lod();
  if (param.Y) {
    if (param.Y->lod().size()) {
      *lod = param.Y->lod();
    } else {
      const auto* y_data = param.Y->data<int>();
      (*lod).resize(1);
      (*lod)[0].resize(param.Y->numel());
      for (int i = 0; i < param.Y->numel(); i++) {
        (*lod)[0][i] = y_data[i];
      }
    }
  } else {
    (*lod).resize(1);
    for (auto id : param.target_lod) {
      (*lod)[0].push_back(id);
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lod_reset,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::LodResetCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
