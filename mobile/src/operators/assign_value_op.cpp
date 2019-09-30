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

#include "operators/assign_value_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void AssignValueOp<Dtype, T>::InferShape() const {
  const auto &shape = this->param_.shape_;
  this->param_.output_->Resize(framework::make_ddim(shape));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(assign_value, ops::AssignValueOp);
#endif

#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(assign_value, ops::AssignValueOp);
#endif

#endif  // ASSIGN_VALUE_OP
