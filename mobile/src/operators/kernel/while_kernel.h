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

#pragma once

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#ifdef WHILE_OP
template <typename Dtype>
class WhileParam : public OpParam {
 public:
  WhileParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, Scope *scope)
      : scope_(scope), OpParam(inputs, outputs, attrs, scope) {
    cond_ =
        OpParam::GetVarValue<framework::LoDTensor>("Condition", inputs, *scope);
    sub_block_ = OpParam::GetAttr<framework::BlockDesc *>("sub_block", attrs);
    is_test = OpParam::GetAttr<bool>("is_test", attrs);
  }

 public:
  Scope *scope_;
  framework::LoDTensor *cond_;
  framework::BlockDesc *sub_block_;
  bool is_test;
};

DECLARE_KERNEL(While, WhileParam);
#endif  // WHILE_OP

}  // namespace operators
}  // namespace paddle_mobile
