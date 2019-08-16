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

#pragma once

#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype>
class ConditionalBlockParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ConditionalBlockParam(const VariableNameMap &inputs,
                        const VariableNameMap &outputs,
                        const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    input_ = OpParam::GetMultiVarValue<GType>("Input", inputs, *scope);
    cond_ = OpParam::GetMultiVarValue<GType>("Cond", inputs, *scope);
    output_ = OpParam::OutFrom<GType>(outputs, *scope);
    scope_ = OpParam::GetVar("Scope", outputs, *scope);
    is_scalar_condition_ = GetAttr<bool>("is_scalar_condition", attrs);
    sub_block_ = GetAttr<framework::BlockDesc *>("sub_block", attrs);
  }

  const vector<GType *> Input() const { return input_; }

  const vector<GType *> Cond() const { return cond_; }

  GType *Output() const { return output_; }

  Variable *OutputScope() const { return scope_; }

  bool isScalarCondition() const { return is_scalar_condition_; }

  framework::BlockDesc *getSubBlock() const { return sub_block_; }

 private:
  vector<GType *> input_;
  vector<GType *> cond_;
  GType *output_;
  Variable *scope_;
  bool is_scalar_condition_;
  framework::BlockDesc *sub_block_;
};

DECLARE_KERNEL(ConditionalBlock, ConditionalBlockParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // CONDITIONAL_BLOCK_OP
