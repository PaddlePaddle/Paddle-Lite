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

#include "operators/kernel/assign_value_kernel.h"
#include "framework/data_type.h"

namespace paddle_mobile {
namespace operators {

struct AssignValueOpFunctor {
  framework::LoDTensor* output_;
  const std::vector<int> shape_;
  const std::vector<int> int32_values_;
  const std::vector<float> fp32_values_;

  AssignValueOpFunctor(framework::LoDTensor* output,
                       const std::vector<int>& shape,
                       const std::vector<float>& fp32_values,
                       const std::vector<int>& int32_values)
      : output_(output),
        shape_(shape),
        int32_values_(int32_values),
        fp32_values_(fp32_values) {}

  template <typename T>
  inline void apply() const {
    PADDLE_MOBILE_THROW_EXCEPTION("Assign value: not supported data type.");
  }
};

template <>
inline void AssignValueOpFunctor::apply<int>() const {
  framework::TensorFromVector<int>(int32_values_, output_);
  output_->Resize(framework::make_ddim(shape_));
}

template <>
inline void AssignValueOpFunctor::apply<float>() const {
  framework::TensorFromVector<float>(fp32_values_, output_);
  output_->Resize(framework::make_ddim(shape_));
}

template <>
bool AssignValueKernel<CPU, float>::Init(AssignValueParam<CPU>* param) {
  return true;
}

template <>
void AssignValueKernel<CPU, float>::Compute(
    const AssignValueParam<CPU>& param) {
  framework::VisitDataType(
      framework::ToDataType(param.dtype_),
      AssignValueOpFunctor(param.output_, param.shape_, param.fp32_values_,
                           param.int32_values_));
}

template <>
bool AssignValueKernel<GPU_CL, float>::Init(AssignValueParam<GPU_CL>* param) {
  return true;
}

template <>
void AssignValueKernel<GPU_CL, float>::Compute(
    const AssignValueParam<GPU_CL>& param) {
  framework::VisitDataType(
      framework::ToDataType(param.dtype_),
      AssignValueOpFunctor(param.output_, param.shape_, param.fp32_values_,
                           param.int32_values_));
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ASSIGN_VALUE_OP
