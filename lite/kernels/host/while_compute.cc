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

#include "lite/kernels/host/while_compute.h"
#include <unordered_map>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void WhileCompute::PrepareForRun() {
  auto &param = this->Param<param_t>();
  program_.reset(new RuntimeProgram(
      param.program_desc, param.exec_scope, param.block_idx));
}
void WhileCompute::Run() {
  auto &param = this->Param<param_t>();
  while (param.cond->data<bool>()[0]) {
    program_->Run();
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    while, kHost, kAny, kAny, paddle::lite::kernels::host::WhileCompute, def)
    .BindInput("X",
               {LiteType::GetTensorListTy(
                   TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindInput("Condition",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("StepScopes",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();
