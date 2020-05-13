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
#include "lite/operators/conditional_block_op.h"
#include "lite/operators/subgraph_op.h"
#include "lite/operators/while_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void StepExecutor::Build() {
  CHECK(block_idx_ >= 0 && block_idx_ < program_desc_->BlocksSize());
  auto *block_desc = program_desc_->GetBlock<cpp::BlockDesc>(block_idx_);
  for (int op_idx = 0; op_idx < block_desc->OpsSize(); op_idx++) {
    auto *op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    if (op_type == "feed" || op_type == "fetch") continue;
    // Create op and pick up the best kernel
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    if (op_type == "while") {
      static_cast<operators::WhileOpLite *>(op.get())->SetProgramDesc(
          program_desc_);
    } else if (op_type == "conditional_block") {
      /*static_cast<operators::ConditionalBlockOpLite*>(op.get())
          ->SetProgramDesc(program_desc);*/
    } else if (op_type == "subgraph") {
      static_cast<operators::SubgraphOp *>(op.get())->SetProgramDesc(
          program_desc_);
    }
    op->Attach(*op_desc, scope_);
    std::unique_ptr<KernelBase> picked_kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      // Create op and pick up the best kernel according to the
      // kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "Found the attr '" << kKernelTypeAttr << "': " << kernel_type
              << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase> &it) {
            return it->alias() == alias;
          });
      CHECK(it != kernels.end());
      picked_kernel = std::move(*it);
    } else {
      VLOG(3) << "The attr '" << kKernelTypeAttr
              << "' not found, pick the first kernel for " << op_type;
      std::vector<std::unique_ptr<KernelBase>> kernels;
#if defined(LITE_WITH_ARM)
      kernels = op->CreateKernels({Place{TARGET(kARM)}, Place{TARGET(kHost)}});
#elif defined(LITE_WITH_X86)
      kernels = op->CreateKernels({Place{TARGET(kX86)}, Place{TARGET(kHost)}});
#endif
      if (kernels.size() > 0) {
        picked_kernel = std::move(kernels.front());
      } else {
        LOG(WARNING) << "No kernels found for " << op_type;
      }
    }
    picked_kernel->SetContext(
        ContextScheduler::Global().NewContext(picked_kernel->target()));
    insts_.emplace_back(std::move(op), std::move(picked_kernel));
  }
}

void StepExecutor::Run() {
  if (!insts_.empty()) {
    for (auto &inst : insts_) {
      auto op_type = inst.op()->op_info()->Type();
      if (op_type == "feed" || op_type == "fetch") continue;
      inst.Run();
    }
  } else {
    Build();
  }
}

void WhileCompute::PrepareForRun() {
  auto &param = this->Param<param_t>();
  executor_ = std::make_shared<StepExecutor>(
      param.block_idx, param.program_desc, param.scope);
  executor_->Build();
}

void WhileCompute::Run() {
  auto &param = this->Param<param_t>();
  while (param.cond->data<bool>()[0]) {
    executor_->Run();
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
