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

#include "operators/kernel/one_hot_kernel.h"
#include "framework/data_type.h"

namespace paddle_mobile {
namespace operators {

template <typename InT>
struct OnehotOpFunctor {
  const framework::LoDTensor* in_;
  framework::LoDTensor* out_;
  int depth_;

  OnehotOpFunctor(const framework::LoDTensor* in, framework::LoDTensor* out,
                  int depth)
      : in_(in), out_(out), depth_(depth) {}

  template <typename OutT>
  void apply() const {
    auto* p_in_data = in_->data<InT>();
    auto numel = in_->numel();
    auto* p_out_data = out_->mutable_data<OutT>();
    memset(p_out_data, 0, out_->numel() * sizeof(OutT));

    for (int i = 0; i < numel; ++i) {
      *(p_out_data + i * depth_ + p_in_data[i]) = 1.0;
    }
  }
};

template <>
bool OnehotKernel<CPU, float>::Init(OnehotParam<CPU>* param) {
  return true;
}

template <>
void OnehotKernel<CPU, float>::Compute(const OnehotParam<CPU>& param) {
  framework::VisitDataType(
      framework::ToDataType(param.dtype_),
      OnehotOpFunctor<int64_t>(param.input_, param.output_, param.depth_));
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ONE_HOT_OP
