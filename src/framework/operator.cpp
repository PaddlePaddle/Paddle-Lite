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

#include "framework/operator.h"
#include "framework/op_info.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
OperatorBase<Dtype>::OperatorBase(const std::string &type,
                                  const VariableNameMap &inputs,
                                  const VariableNameMap &outputs,
                                  const AttributeMap &attrs,
                                  std::shared_ptr<Scope> scope)
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      scope_(scope) {
  CheckAllInputOutputSet();
}
template <typename Dtype>
void OperatorBase<Dtype>::CheckAllInputOutputSet() const {}

template class OperatorBase<CPU>;
template class OperatorWithKernel<CPU>;

}  // namespace framework
}  // namespace paddle_mobile
