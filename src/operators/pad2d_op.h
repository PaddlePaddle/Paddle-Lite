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

#ifdef PAD2D_OP

#pragma once

#include <string>

#include "framework/operator.h"
#include "operators/kernel/pad2d_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
using framework::AttributeMap;
using framework::OperatorWithKernel;
using framework::Scope;
using std::string;
template <typename DeviceType, typename T>
class Pad2dOp
    : public OperatorWithKernel<DeviceType, Pad2dParam<DeviceType>,
                                operators::Pad2dKernel<DeviceType, T>> {
 public:
  Pad2dOp(const string &type, const VariableNameMap &inputs,
          const VariableNameMap &outputs, const AttributeMap &attrs,
          std::shared_ptr<Scope> scope)
      : OperatorWithKernel<DeviceType, Pad2dParam<DeviceType>,
                           operators::Pad2dKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}
  void InferShape() const override;

 private:
};
}  // namespace operators
}  // namespace paddle_mobile

#endif
