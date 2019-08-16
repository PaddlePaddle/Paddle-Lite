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

#include "operators/range_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void RangeOp<Dtype, T>::InferShape() const {
  auto s_dims = this->param_.Start()->dims();
  PADDLE_MOBILE_ENFORCE((s_dims.size() == 1) && (s_dims[0] == 1),
                        "The shape of Input(Start) should be [1].");
  auto e_dims = this->param_.End()->dims();
  PADDLE_MOBILE_ENFORCE((e_dims.size() == 1) && (e_dims[0] == 1),
                        "The shape of Input(End) should be [1].");
  auto step_dims = this->param_.Step()->dims();
  PADDLE_MOBILE_ENFORCE((step_dims.size() == 1) && (step_dims[0] == 1),
                        "The shape of Input(Step) should be [1].");
  this->param_.Output()->Resize({-1});
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(range, ops::RangeOp);
#endif

#endif  // ASSIGN_OP
