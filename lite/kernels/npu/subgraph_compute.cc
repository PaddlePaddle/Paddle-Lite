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
  auto sub_block = param.sub_block;
  CHECK(sub_block);
  auto scope = param.scope;
  CHECK(scope);
  int32_t ops_size = sub_block->OpsSize();
  std::vector<Instruction> insts;
  for (int32_t i = 0; i < ops_size; i++) {
    auto& op_desc = *sub_block->GetOp<cpp::OpDesc>(i);
    auto kernel_type = op_desc.GetAttr<std::string>(lite::kKernelTypeAttr);
    // Create op and pick up kernel according to the kKernelTypeAttr attribute
    auto op = lite::LiteOpRegistry::Global().Create(op_desc.Type());
    op->Attach(op_desc, scope);
    std::string op_type, alias;
    Place place;
    KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
    LOG(INFO) << "op_type: " << op_type << " kernel_type: " << kernel_type;
    auto kernels = op->CreateKernels({place});
    CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
    auto it = std::find_if(
        kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& it) {
          return it->alias() == alias;
        });
    CHECK(it != kernels.end());
    (*it)->SetContext(ContextScheduler::Global().NewContext((*it)->target()));
    insts.emplace_back(std::move(op), std::move(*it));
  }
  sub_program_.reset(new RuntimeProgram(std::move(insts)));
  sub_program_->set_exec_scope(scope);
}

void SubgraphCompute::Run() {
  auto& param = this->Param<param_t>();
  sub_program_->Run();
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
