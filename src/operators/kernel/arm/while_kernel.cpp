/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef WHILE_OP

#include "operators/kernel/while_kernel.h"
#include "framework/op_registry.h"
#include "framework/operator.h"

namespace paddle_mobile {
namespace operators {

class StepExecutor {
  typedef std::shared_ptr<framework::OperatorBase<CPU>> OperatorPtr;

 public:
  StepExecutor(const framework::BlockDesc *block, framework::Scope *scope)
      : scope_(scope) {
    std::vector<std::shared_ptr<framework::OpDesc>> ops = block->Ops();
    ops_of_block_.resize(ops.size());
    for (int i = 0; i < ops.size(); ++i) {
      std::shared_ptr<framework::OpDesc> op_desc = ops[i];
      DLOG << "create op: " << op_desc->Type();
      auto op_handler = framework::OpRegistry<CPU>::CreateOp(
          op_desc->Type(), op_desc->GetInputs(), op_desc->GetOutputs(),
          op_desc->GetAttrMap(), scope_);
      op_handler->Init();
      ops_of_block_[i] = op_handler;
    }
  }

  void Run() {
    for (auto &op_handler : ops_of_block_) {
      op_handler->InferShape();
      op_handler->Run();
    }
  }

 private:
  framework::Scope *scope_;
  std::vector<OperatorPtr> ops_of_block_;
};

template <>
bool WhileKernel<CPU, float>::Init(WhileParam<CPU> *param) {
  return true;
}

template <>
void WhileKernel<CPU, float>::Compute(const WhileParam<CPU> &param) {
  auto &current_scope = param.scope_->NewScope();
  StepExecutor executor(param.sub_block_, &current_scope);
  while (param.cond_->data<bool>()[0]) {
    executor.Run();
  }
  param.scope_->DeleteScope(&current_scope);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // WHILE_OP
