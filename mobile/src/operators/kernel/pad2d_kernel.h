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
#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

// template <typename Dtype>
// class Pad2DParam : public OpParam {
// public:
//  Pad2DParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
//             const AttributeMap &attrs, Scope *scope)
//      : OpParam(inputs, outputs, attrs, scope) {
//    input_ = OpParam::GetVarValue<framework::LoDTensor>("X", inputs, *scope);
//    output_ =
//        OpParam::GetVarValue<framework::LoDTensor>("Out", outputs, *scope);
//    paddings_ = OpParam::GetAttr<std::vector<int>>("paddings", attrs);
//    pad_value_ = OpParam::GetAttr<float>("pad_value", attrs);
//    mode_ = OpParam::GetStringAttr("mode", attrs);
//  }
//
// public:
//  framework::LoDTensor *input_;
//  framework::LoDTensor *output_;
//  std::vector<int> paddings_;
//  float pad_value_;
//  std::string mode_;
//};

DECLARE_KERNEL(Pad2D, Pad2DParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PAD2D_OP
