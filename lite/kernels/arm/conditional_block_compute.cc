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

#include "lite/kernels/arm/conditional_block_compute.h"
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

void ConditionalBlockCompute::PrepareForRun() {
  auto& param = Param<operators::ConditionalBlockParam>();
  auto cur_scope = param.scope;

  executor_ =
      std::make_shared<CondExecutor>(param.sub_block, cur_scope, place());
}
void ConditionalBlockCompute::Run() {
  auto& param = Param<operators::ConditionalBlockParam>();
  for (auto& out : param.outs) {
    out->clear();
  }
  bool need_run = true;
  if (param.is_scalar_condition) {
    auto* cond = param.cond;
    auto* cond_data = cond->data<bool>();
    need_run = cond_data[0];
  } else {
    auto x = param.x;
    for (auto pt : x) {
      if (pt == nullptr || !pt->IsInitialized() || pt->dims().empty()) {
        need_run = false;
        break;
      }
    }
  }
  if (need_run) {
    executor_->Run();
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conditional_block,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ConditionalBlockCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Cond", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Scope", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
