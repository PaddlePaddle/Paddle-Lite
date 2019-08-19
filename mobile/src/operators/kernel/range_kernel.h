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

#ifdef RANGE_OP

#pragma once

#include <cmath>
#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

inline void GetSize(float start, float end, float step, int64_t *size) {
  PADDLE_MOBILE_ENFORCE(!std::equal_to<float>()(step, 0),
                        "The step of range op should not be 0.");
  PADDLE_MOBILE_ENFORCE(
      ((start < end) && (step > 0)) || ((start > end) && (step < 0)),
      "The step should be greater than 0 while start < end. And the "
      "step should be less than 0 while start > end.");
  *size = std::is_integral<float>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

template <typename Dtype>
class RangeParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  RangeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    start_ = OpParam::GetVarValue<GType>("Start", inputs, *scope);
    end_ = OpParam::GetVarValue<GType>("End", inputs, *scope);
    step_ = OpParam::GetVarValue<GType>("Step", inputs, *scope);
    output_ = OpParam::OutFrom<GType>(outputs, *scope);
  }

  GType *Start() const { return start_; }
  const GType *End() const { return end_; }
  const GType *Step() const { return step_; }
  GType *Output() const { return output_; }

 private:
  GType *start_;
  GType *end_;
  GType *step_;
  GType *output_;
};

DECLARE_KERNEL(Range, RangeParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // RANGE_OP
