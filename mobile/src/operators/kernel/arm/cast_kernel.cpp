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

#ifdef CAST_OP

#include <algorithm>
#include <vector>
#include "framework/data_type.h"
#include "operators/kernel/kernels.h"

namespace paddle_mobile {
namespace operators {

template <typename InT>
struct CastOutOpFunctor {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  CastOutOpFunctor(const framework::Tensor* in, framework::Tensor* out)
      : in_(in), out_(out) {}

  template <typename OutT>
  void apply() const {
    const InT* input = in_->data<InT>();
    OutT* output = out_->mutable_data<OutT>();
    size_t numel = in_->numel();
    for (int i = 0; i < numel; ++i) {
      output[i] = static_cast<OutT>(input[i]);
    }
  }
};

// struct CastOpFunctor {
//  const framework::Tensor* in_;
//  framework::Tensor* out_;
//  int output_type_;
//  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
//                const int output_type)
//      : in_(in), out_(out), output_type_(output_type) {}
//
//  template <typename InT>
//  void apply() const {
//    framework::VisitDataType(framework::ToDataType(output_type_),
//                             CastOutOpFunctor<InT>(in_, out_));
//  }
//};

template <>
bool CastKernel<CPU, float>::Init(CastParam<CPU>* param) {
  return true;
}

template <>
void CastKernel<CPU, float>::Compute(const CastParam<CPU>& param) {
  const Tensor* input = param.input_;
  Tensor* output = param.output_;
  if (input->type() == type_id<float>()) {
    framework::VisitDataType(framework::ToDataType(param.output_type_),
                             CastOutOpFunctor<float>(input, output));
  } else if (input->type() == type_id<int64_t>()) {
    framework::VisitDataType(framework::ToDataType(param.output_type_),
                             CastOutOpFunctor<int64_t>(input, output));
  } else if (input->type() == type_id<int>()) {
    framework::VisitDataType(framework::ToDataType(param.output_type_),
                             CastOutOpFunctor<int>(input, output));
  } else {
    PADDLE_MOBILE_ENFORCE(0, "input tpye not support now!")
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // CAST_OP
