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
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace arena {

void TestCase::CreateInstruction() {
  std::shared_ptr<lite::OpLite> op = nullptr;
  if (place_.target == TARGET(kNPU) || place_.target == TARGET(kXPU)) {
    // Create a new block desc to wrap the original op desc
    int sub_block_idx = 0;
    cpp::BlockDesc sub_block_desc;
    sub_block_desc.ClearOps();
    sub_block_desc.ClearVars();
    auto sub_block_op_desc = sub_block_desc.AddOp<cpp::OpDesc>();
    *sub_block_op_desc = *op_desc_;
    // Add the block desc into the subgraph op which used to replace the
    // original op
    op_desc_.reset(new cpp::OpDesc());
    op_desc_->SetType("subgraph");
    op_desc_->SetAttr<int32_t>("sub_block", sub_block_idx);
    op_desc_->SetInput("Inputs", op_desc_->input_vars());
    op_desc_->SetOutput("Outputs", op_desc_->output_vars());
    op_desc_->SetAttr<std::vector<std::string>>(
        "input_data_names", sub_block_op_desc->input_vars());
    op_desc_->SetAttr<std::vector<std::string>>(
        "output_data_names", sub_block_op_desc->output_vars());
    op = LiteOpRegistry::Global().Create(op_desc().Type());
    static_cast<operators::SubgraphOp*>(op.get())->SetSubBlock(sub_block_desc);
  } else {
    op = LiteOpRegistry::Global().Create(op_desc().Type());
  }
  CHECK(op) << "no op for " << op_desc().Type();
  op->Attach(*op_desc_, inst_scope_);
  auto kernels = op->CreateKernels({place_});
  // filter out the target kernel
  CHECK(!kernels.empty()) << "No kernel found for place "
                          << place_.DebugString();
  auto it = std::remove_if(
      kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& k) {
        return k->alias() == alias_;
      });
  CHECK(it != kernels.end()) << "failed to create the kernel in "
                             << place_.DebugString()
                             << " with alias: " << alias_;
  // prepare context
  (*it)->SetContext(std::move(ctx_));
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
