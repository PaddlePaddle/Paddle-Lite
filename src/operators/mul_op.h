/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "framework/operator.h"
#include "operators/kernel/mul_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
    namespace operators {

        using namespace framework;

        template <typename DeviceType, typename T>
        class MulOp : public framework::OperatorWithKernel<DeviceType> {
          public:
            MulOp(const std::string &type, const VariableNameMap &inputs,
                  const VariableNameMap &outputs,
                  const framework::AttributeMap attrs,
                  std::shared_ptr<framework::Scope> scope)
                : framework::OperatorWithKernel<DeviceType>(
                      type, inputs, outputs, attrs, scope),
                  param_(inputs, outputs, attrs, *scope) {}

            void Run() const {
                operators::MulKernel<DeviceType, T, MulParam> kernel;
                kernel.Compute(param_);
            }

            using framework::OperatorWithKernel<DeviceType>::OperatorWithKernel;
            void InferShape() const override;

          protected:
            MulParam param_;
        };

    } // namespace operators
} // namespace paddle
