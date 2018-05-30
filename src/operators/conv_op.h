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

#include <string>
#include "framework/operator.h"
#include "operators/kernel/conv_kernel.h"

namespace paddle_mobile {
namespace operators {
using std::string;
template <typename DeviceType, typename T>
class ConvOp : public framework::OperatorWithKernel<DeviceType> {
 public:
  ConvOp(const std::string &type, const VariableNameMap &inputs,
         const VariableNameMap &outputs, const framework::AttributeMap &attrs,
         std::shared_ptr<framework::Scope> scope)
      : framework::OperatorWithKernel<DeviceType>(type, inputs, outputs, attrs,
                                                  scope),
        param_(inputs, outputs, attrs, *scope) {}

  using framework::OperatorWithKernel<DeviceType>::OperatorWithKernel;
  void InferShape() const override;

  void RunImpl() const {
    operators::ConvKernel<DeviceType, T> kernel;
    kernel.Compute(param_);
    this->ClearVariables({"Filter", "Input"});
  }

 private:
  ConvParam param_;
};

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

}  // namespace operators
}  // namespace paddle_mobile
