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

#include "lite/kernels/arm/stack_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void StackCompute::Run() {
  auto& param = Param<operators::StackParam>();
  std::vector<lite::Tensor*> x = param.X;
  lite::Tensor* out = param.Out;
  int axis = param.axis;

  lite::arm::math::stack(x, out, axis);
}

} /* namespace arm */
} /* namespace kernels */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_KERNEL(
    stack, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::StackCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
