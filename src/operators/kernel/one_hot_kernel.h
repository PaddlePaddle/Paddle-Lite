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

#ifdef ONE_HOT_OP

#pragma once

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#define GET_VAR_AS_LOD_TENSOR(name, name_dict, scope) \
  OpParam::GetVarValue<framework::LoDTensor>(name, name_dict, scope)

template <typename Dtype>
class OnehotParam : public OpParam {
 public:
  OnehotParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    input_ = GET_VAR_AS_LOD_TENSOR("X", inputs, *scope);
    output_ = GET_VAR_AS_LOD_TENSOR("Out", outputs, *scope);

    depth_ = OpParam::GetAttr<int>("depth", attrs);
    dtype_ = OpParam::GetAttr<int>("dtype", attrs);
  }

 public:
  framework::LoDTensor *input_;
  framework::LoDTensor *output_;

  int depth_;
  int dtype_;
};

DECLARE_KERNEL(Onehot, OnehotParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ONE_HOT_OP
