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

#ifdef ASSIGN_VALUE_OP

#pragma once

#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype>
class AssignValueParam : public OpParam {
 public:
  AssignValueParam(const VariableNameMap &inputs,
                   const VariableNameMap &outputs, const AttributeMap &attrs,
                   Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    output_ = GET_VAR_AS_LOD_TENSOR("Out", outputs, *scope);
    shape_ = OpParam::GetAttr<std::vector<int>>("shape", attrs);
    fp32_values_ = OpParam::GetAttr<std::vector<float>>("fp32_values", attrs);
    int32_values_ = OpParam::GetAttr<std::vector<int>>("int32_values", attrs);
    dtype_ = OpParam::GetAttr<int>("dtype", attrs);
  }

 public:
  framework::LoDTensor *output_;
  std::vector<int> shape_;
  std::vector<float> fp32_values_;
  std::vector<int> int32_values_;
  int dtype_;
};

DECLARE_KERNEL(AssignValue, AssignValueParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ASSIGN_VALUE_OP
