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

#include "lite/kernels/fpga/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void SubgraphCompute::PrepareForRun() {
  VLOG(3) << "preparing fpga subgraph kernel";
  auto &param = this->Param<param_t>();
  program_.reset(new RuntimeProgram(
      param.program_desc, param.exec_scope, param.block_idx));
}

void SubgraphCompute::Run() {
  VLOG(3) << "Running fpga subgraph kernel";
  program_->Run();
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kFPGA,
                     kAny,
                     kNHWC,
                     paddle::lite::kernels::fpga::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
