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
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
using std::string;

template <typename DeviceType, typename T>
class FetchOp : public framework::OperatorBase<DeviceType> {
 public:
  FetchOp(const string &type, const VariableNameMap &inputs,
          const VariableNameMap &outputs, const framework::AttributeMap attrs,
          std::shared_ptr<framework::Scope> scope)
      : framework::OperatorBase<DeviceType>(type, inputs, outputs, attrs,
                                            scope),
        param_(inputs, outputs, attrs, *scope) {}
  void RunImpl() const { param_.Out()->ShareDataWith(*param_.InputX()); }

  void Init() {}

  void InferShape() const {
    auto x_dims = param_.InputX()->dims();
    param_.Out()->Resize(x_dims);
  }

 protected:
  FetchParam<DeviceType> param_;
};

}  // namespace operators
}  // namespace paddle_mobile

#ifdef PADDLE_MOBILE_CPU
USE_OP_CPU(fetch);
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
USE_OP_MALI_GPU(fetch);
#endif
#ifdef PADDLE_MOBILE_FPGA
USE_OP_FPGA(fetch);
#endif
