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

#ifdef ASSIGN_OP

#pragma once

#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype>
class AssignParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  AssignParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    input_ = OpParam::InputXFrom<GType>(inputs, *scope);
    output_ = OpParam::OutFrom<GType>(outputs, *scope);
  }

  const GType *Input() const { return input_; }

  GType *Output() const { return output_; }

 private:
  GType *input_;
  GType *output_;
};

DECLARE_KERNEL(Assign, AssignParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ASSIGN_OP
