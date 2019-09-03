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

#include "lite/kernels/arm/while_compute.h"
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void WhileCompute::PrepareForRun() {
  auto &param = Param<operators::WhileParam>();
  auto cur_scope = param.scope;

  executor_ =
      std::make_shared<StepExecutor>(param.sub_block, cur_scope, place());
}
void WhileCompute::Run() {
  auto &param = Param<operators::WhileParam>();
  while (param.cond->data<bool>()[0]) {
    executor_->Run();
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    while, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::WhileCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Condition",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorListTy(TARGET(kARM))})
    .BindOutput("StepScopes", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
