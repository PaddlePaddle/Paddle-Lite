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

#include <framework/operator.h>
#include <operators/op_param.h>
#include <string>
#include "operators/kernel/softmax_kernel.h"
#if defined(USE_ACL)
#include "operators/kernel/mali/acl_softmax_op.h"
#endif

namespace paddle_mobile {
namespace operators {
template <typename DeviceType, typename T>
class SoftmaxOp : public framework::OperatorWithKernel<DeviceType> {
 public:
  SoftmaxOp(const std::string &type, const VariableNameMap &inputs,
            const VariableNameMap &outputs,
            const framework::AttributeMap &attrs,
            std::shared_ptr<framework::Scope> scope)
      : framework::OperatorWithKernel<DeviceType>(type, inputs, outputs, attrs,
                                                  scope),
        param_(inputs, outputs, attrs, *scope) {}

  using framework::OperatorWithKernel<DeviceType>::OperatorWithKernel;

  void InferShape() const override;

  void RunImpl() const {
#if defined(USE_ACL)
    std::cout << "Using ACL!" << std::endl;
    if (std::is_same<T, float>::value &&
        !acl_softmax_kernel_.Bypass_acl(param_)) {
      acl_softmax_kernel_.Compute(param_);
      this->ClearVariables({"X"});
      return;
    }
#endif
    std::cout << "Not using ACL!" << std::endl;
    operators::SoftmaxKernel<DeviceType, T> kernel;
    kernel.Compute(param_);
    this->ClearVariables({"X"});
  }

 private:
  SoftmaxParam param_;
#if defined(USE_ACL)
  AclSoftmaxKernel<DeviceType, T> acl_softmax_kernel_;
#endif
};
}  // namespace operators
}  // namespace paddle_mobile
