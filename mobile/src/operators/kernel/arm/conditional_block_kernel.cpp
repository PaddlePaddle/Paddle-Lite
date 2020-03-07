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

#ifdef CONDITIONAL_BLOCK_OP

#include "operators/kernel/conditional_block_kernel.h"
#include <framework/program/block_desc.h>
#include <framework/program/op_desc.h>
#include <algorithm>
#include "framework/data_type.h"

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
      DLOG << "conditional block create op: " << ops.size() << ","
           << op_desc->Type();
      auto op_handler = framework::OpRegistry<CPU>::CreateOp(
          op_desc->Type(), op_desc->GetInputs(), op_desc->GetOutputs(),
          op_desc->GetAttrMap(), scope_);
      op_handler->Init();
      ops_of_block_[i] = op_handler;
    }
  }

  void Run() {
    for (int i = 0; i < ops_of_block_.size(); ++i) {
      auto &op_handler = ops_of_block_[i];
      DLOG << "conditional block op InferShape: " << i
           << "th: " << op_handler->Type();
      op_handler->InferShape();
      DLOG << "conditional block op Run: " << i << "th: " << op_handler->Type();
      op_handler->Run();
    }
  }

 private:
  framework::Scope *scope_;
  std::vector<OperatorPtr> ops_of_block_;
};

template <>
bool ConditionalBlockKernel<CPU, float>::Init(
    ConditionalBlockParam<CPU> *param) {
  return true;
}

template <>
void ConditionalBlockKernel<CPU, float>::Compute(
    const ConditionalBlockParam<CPU> &param) {
  bool need_run;
  if (param.isScalarCondition()) {
    auto xs = param.Cond();
    PADDLE_MOBILE_ENFORCE(
        xs[0]->type() == type_id<bool>().hash_code() && xs[0]->numel() == 1,
        "condition input's data type should be bool, "
        "numel should be 1, actual numel is %d",
        xs[0]->numel());
    need_run = xs[0]->data<bool>()[0];
  } else {
    auto xs = param.Input();
    need_run = std::all_of(
        xs.begin(), xs.end(),
        [](const framework::LoDTensor *t) { return t->numel() != 0; });
  }

  if (need_run) {
    auto input = param.Input();
    auto sub = param.getSubBlock();
    auto &current_scope = param.GetScope()->NewScope();
    StepExecutor executor(sub, &current_scope);
    executor.Run();
    param.GetScope()->DeleteScope(&current_scope);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // CONDITIONAL_BLOCK_OP
