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

#ifdef ELEMENTWISEADD_OP

#include "elementwise_add_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ElementwiseAddOp<Dtype, T>::InferShape() const {
  auto x_dim = param_.InputX()->dims();
  param_.Out()->Resize(x_dim);
}
template class ElementwiseAddOp<CPU, float>;
}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
USE_OP_CPU(elementwise_add);
REGISTER_OPERATOR_CPU(elementwise_add, ops::ElementwiseAddOp);
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif

#endif
