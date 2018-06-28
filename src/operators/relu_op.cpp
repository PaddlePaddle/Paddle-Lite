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

#ifdef RELU_OP

#include "operators/relu_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ReluOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.InputX()->dims();
  this->param_.Out()->Resize(input_dims);
}
template class ReluOp<CPU, float>;
}  // namespace operators
}  // namespace paddle_mobile

/*
 * @b 每一个 op 都需要注册一下的,
 *    USE_OP的参数 和 REGISTER_OPERATOR的第一个参数
 * 都是需要和model中类型对应起来的
 * */
namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
USE_OP_CPU(relu);
REGISTER_OPERATOR_CPU(relu, ops::ReluOp);
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
<<<<<<< HEAD
USE_OP_MALI_GPU(relu);
REGISTER_OPERATOR_MALI_GPU(relu, ops::ReluOp);
=======
>>>>>>> c71c2f8879fc105d1d144df744a5dfef3ab2a77b
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif

#endif
