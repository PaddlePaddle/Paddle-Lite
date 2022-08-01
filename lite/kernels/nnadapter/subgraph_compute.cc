// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/nnadapter/subgraph_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(param.program_desc.get());
  auto block_count = param.program_desc->BlocksSize();
  auto block_index = param.block_idx;
  CHECK_GT(block_count, 0) << "No block found!";
  CHECK_LT(block_index, block_count) << "Invalid block index, expected [0,"
                                     << (block_count - 1) << "] but recieved "
                                     << block_index;
  auto block_desc = param.program_desc->GetBlock<cpp::BlockDesc>(block_index);
  CHECK(block_desc);
  engine_.reset(new Engine(ctx_.get(),
                           block_desc,
                           param.exec_scope,
                           param.input_data_names,
                           param.output_data_names,
                           param.input_data_scales,
                           param.output_data_scales));
  CHECK(engine_);
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Run();
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kNNAdapter,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::nnadapter::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kAny),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
