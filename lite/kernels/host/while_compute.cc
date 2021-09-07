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
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

bool GetCondData(const Tensor *cond) {
  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM);
  };

  bool flag;
  if (is_host(cond->target())) {
    flag = cond->data<bool>()[0];
  } else if (cond->target() == TARGET(kXPU)) {
#ifdef LITE_WITH_XPU
    TargetWrapperXPU::MemcpySync(
        &flag, cond->raw_data(), cond->memory_size(), IoDirection::DtoH);
#endif
  } else {
    LOG(ERROR) << "Unsupported target: "
               << lite_api::TargetToStr(cond->target());
  }
  return flag;
}

void WhileCompute::PrepareForRun() {
  auto &param = this->Param<param_t>();
  if (program_ == nullptr) {
    program_.reset(new RuntimeProgram(
        param.program_desc, param.exec_scope, param.block_idx));
  }
}

void WhileCompute::Run() {
  auto &param = this->Param<param_t>();
  auto cond = param.cond;
  while (GetCondData(cond)) {
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
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Condition",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("StepScopes", {LiteType::GetStepScopeTy()})
    .Finalize();
