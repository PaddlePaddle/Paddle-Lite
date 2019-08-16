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

#ifdef PRELU_OP

#include "operators/prelu_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void PReluOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.InputX()->dims();
  this->param_.Out()->Resize(input_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

/*
 * @b 每一个 op 都需要注册一下的,
 *    USE_OP的参数 和 REGISTER_OPERATOR的第一个参数
 * 都是需要和model中类型对应起来的
 * */
namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(prelu, ops::PReluOp);
#endif

#endif
