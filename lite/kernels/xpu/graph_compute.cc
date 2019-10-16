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

#include "lite/kernels/xpu/graph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GraphCompute::PrepareForRun() {
  // auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->Param<param_t>();

  CHECK(param.weight);

  for (size_t i = 0; i < param.inputs.size(); i++) {
    VLOG(3) << "input dims[" << i << "]: " << param.inputs[i]->dims();
  }
  for (size_t i = 0; i < param.outputs.size(); i++) {
    VLOG(3) << "output dims[" << i << "]: " << param.outputs[i]->dims();
  }
}

void GraphCompute::Run() {
  // auto& param = this->Param<param_t>();
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  // TODO(hong19860320)
  LOG(INFO) << "[XPU] Process cost " << GetCurrentUS() - start_time << " us";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(graph_op,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
