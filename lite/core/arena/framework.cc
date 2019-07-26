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

#include "lite/core/arena/framework.h"
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace arena {

void TestCase::CreateInstruction() {
  LOG(INFO) << "Create op for " << op_desc().Type();
  auto op = LiteOpRegistry::Global().Create(op_desc().Type());
  CHECK(op) << "no op for " << op_desc().Type();
  op->Attach(*op_desc_, inst_scope_);
  auto kernels = op->CreateKernels({place_});
  // filter out the target kernel
  CHECK(!kernels.empty()) << "No kernel found for place " << place_;
  auto it = std::remove_if(
      kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& k) {
        return k->alias() == alias_;
      });
  CHECK(it != kernels.end()) << "failed to create the kernel in " << place_
                             << " with alias: " << alias_;
  // prepare context
  (*it)->SetContext(ContextScheduler::Global().NewContext(place_.target));
  instruction_.reset(new Instruction(op, std::move(*it)));
}

void TestCase::PrepareInputsForInstruction() {
  for (auto& arg : op_desc().InputArgumentNames()) {
    for (auto& var : op_desc().Input(arg)) {
      std::string kernel_key = instruction_->kernel()->key_with_alias();
      const auto* param_type = ParamTypeRegistry::Global().RetrieveInArgument(
          place_, kernel_key, arg);

      const auto* inst_type = Type::GetTensorTy(TARGET(kHost));
      CHECK(scope_->FindVar(var));
      const auto* shared_tensor = scope_->FindTensor((var));
      if (!TargetCompatibleTo(*inst_type, *param_type->type)) {
        /// Create a tensor in the instruction's scope, alloc memory and then
        /// copy data there.
        auto* target_tensor = inst_scope_->NewTensor(var);
        CHECK(!shared_tensor->dims().empty()) << "shared_tensor is empty yet";
        target_tensor->Resize(shared_tensor->dims());
        TargetCopy(param_type->type->target(),
                   target_tensor->mutable_data(param_type->type->target(),
                                               shared_tensor->memory_size()),
                   shared_tensor->raw_data(),
                   shared_tensor->memory_size());
      }
    }
  }
}

}  // namespace arena
}  // namespace lite
}  // namespace paddle
