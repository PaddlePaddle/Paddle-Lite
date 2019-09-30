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

#ifdef INSTANCENORM_OP

#pragma once

#include <string>
#include "framework/operator.h"
#include "operators/kernel/instancenorm_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
using std::string;
template <typename DeviceType, typename T>
class InstanceNormOp
    : public framework::OperatorWithKernel<DeviceType,
                                           InstanceNormParam<DeviceType>,
                                           InstanceNormKernel<DeviceType, T>> {
 public:
  InstanceNormOp(const string &type, const VariableNameMap &inputs,
                 const VariableNameMap &outputs,
                 const framework::AttributeMap &attrs, framework::Scope *scope)
      : framework::OperatorWithKernel<DeviceType, InstanceNormParam<DeviceType>,
                                      InstanceNormKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}

  void InferShape() const override;

 protected:
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
