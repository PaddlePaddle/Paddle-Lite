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

#include "lite/kernels/arm/unbind_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T, PrecisionType PType>
void UnbindCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::UnbindParam>();
  auto& dout = param.output;
  for (auto out : dout) {
    out->set_lod(param.x->lod());
  }
  lite::arm::math::unbind<T>(param.x, dout, param.axis);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using unbind_float =
    paddle::lite::kernels::arm::UnbindCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unbind, kARM, kFloat, kNCHW, unbind_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

using unbind_int64 =
    paddle::lite::kernels::arm::UnbindCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(unbind, kARM, kInt64, kNCHW, unbind_int64, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
