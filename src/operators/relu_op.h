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
#include "operators/kernel/relu_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

using paddle_mobile::framework::Tensor;

template <typename DeviceType, typename T>
class ReluOp : public framework::OperatorWithKernel<DeviceType> {
 public:
  /*
   * @b op 的实例化方法, 需要调用父类的实例化方法, 以及实例化自己的参数结构体
   * */
  ReluOp(const std::string &type, const VariableNameMap &inputs,
         const VariableNameMap &outputs, const framework::AttributeMap attrs,
         std::shared_ptr<framework::Scope> scope)
      : framework::OperatorWithKernel<DeviceType>(type, inputs, outputs, attrs,
                                                  scope),
        param_(inputs, outputs, attrs, *scope) {}

   /*
   * @b op 进行运算, 调用相应的 kernel 进行运算
   * */
  void RunImpl() const {
    operators::ReluKernel<DeviceType, T> kernel;
    kernel.Compute(param_);
  }

  using framework::OperatorWithKernel<DeviceType>::OperatorWithKernel;
  void InferShape() const override;

 protected:
  /*
   * @b Relu kernel 进行运算时所需要用到参数的结构体,
   *    结构体定义在: paddle-mobile/src/operators/op_param.h
   * */
  ReluParam param_;
};

}  // namespace operators
}  // namespace paddle_mobile
