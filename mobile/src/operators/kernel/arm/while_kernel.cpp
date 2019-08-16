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
#include "framework/loader.h"
#include "framework/lod_tensor.h"
#include "framework/op_registry.h"
#include "framework/operator.h"

namespace paddle_mobile {
namespace operators {

class WhileStepExecutor {
  typedef std::shared_ptr<framework::OperatorBase<CPU>> OperatorPtr;

 public:
  WhileStepExecutor(const framework::BlockDesc *block, framework::Scope *scope)
      : scope_(scope) {
    std::vector<std::shared_ptr<framework::OpDesc>> ops = block->Ops();
    ops_of_block_.resize(ops.size());
    for (int i = 0; i < ops.size(); ++i) {
      std::shared_ptr<framework::OpDesc> op_desc = ops[i];
      DLOG << "while kernel create op: " << op_desc->Type();
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
      DLOG << "while kernel InferShape op: " << i
           << "th : " << op_handler->Type();
      op_handler->InferShape();
      DLOG << "while kernel Run op: " << i << "th : " << op_handler->Type();
      op_handler->Run();
    }
  }

  void CreateVariables(Scope &scope, const WhileParam<CPU> &param) {
    for (const auto &var_desc : param.sub_block_->Vars()) {
      auto var = scope.Var(var_desc->Name());
      if (var_desc->Type() == framework::VARTYPE_TYPE_LOD_TENSOR) {
        if (var_desc->Persistable()) {
          auto dim = var_desc->Tensor_desc().Dims();
          auto tensor = var->framework::Variable::GetMutable<LoDTensor>();
          tensor->Resize(framework::make_ddim(dim));
        } else {
          auto dim = var_desc->Tensor_desc().Dims();
          if (dim.size() == 0) {
            auto tensor = var->framework::Variable::GetMutable<LoDTensor>();
            framework::DDim dDim = {0};
            tensor->Resize(dDim);
          } else {
            for (auto &d : dim) {
              if (d < 0) {
                d *= -1;
              }
            }
            auto tensor = var->framework::Variable::GetMutable<LoDTensor>();
            tensor->Resize(framework::make_ddim(dim));
          }
        }
      } else {
        // TODO(codeWorm)
      }
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
  DLOG << "WhileKernel Compute";
  WhileStepExecutor executor(param.sub_block_, param.scope_);
  auto &current_scope = param.scope_->NewScope();
  executor.CreateVariables(current_scope, param);
  while (param.cond_->data<bool>()[0]) {
    if (param.is_test) {
      for (auto &name : current_scope.LocalVarNames()) {
        auto *var = current_scope.Var(name);
        if (var->IsType<framework::LoDTensor>()) {
          // Clear all lod information for all lod_tensors.
          auto *t = var->GetMutable<framework::LoDTensor>();
          framework::LoD empty_lod;
          t->set_lod(empty_lod);
        } else if (var->IsType<framework::LoDTensorArray>()) {
          // Clear elements of all tensor arrays.
          auto *t = var->GetMutable<framework::LoDTensorArray>();
          t->clear();
        } else {
          // todo
        }
      }
    }
    executor.Run();
  }
  param.scope_->DeleteScope(&current_scope);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // WHILE_OP
