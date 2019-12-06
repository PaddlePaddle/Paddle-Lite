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

#include "lite/kernels/npu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

void SubgraphCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<NPUContext>();
  auto& param = this->Param<param_t>();
  engine_.reset(new lite::npu::Engine(param.sub_block_idx,
                                      &param.sub_block_desc,
                                      param.input_data_names,
                                      param.output_data_names,
                                      param.scope));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(engine_);
  engine_->Run();
}

}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kNPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::npu::SubgraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
