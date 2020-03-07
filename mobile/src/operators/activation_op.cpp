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

#include "operators/activation_op.h"

namespace paddle_mobile {
namespace operators {

#define DEFINE_ACTIVATION_INFERSHAPE(OpName)                \
  template <typename Dtype, typename T>                     \
  void OpName##Op<Dtype, T>::InferShape() const {           \
    const auto &input_dims = this->param_.InputX()->dims(); \
    this->param_.Out()->Resize(input_dims);                 \
  }

#ifdef RELU_OP
DEFINE_ACTIVATION_INFERSHAPE(Relu);
DEFINE_ACTIVATION_INFERSHAPE(Relu6);
#endif  // RELU_OP

#ifdef SIGMOID_OP
DEFINE_ACTIVATION_INFERSHAPE(Sigmoid);
namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(sigmoid, ops::SigmoidOp);
#endif
#endif  // SIGMOID_OP

#ifdef TANH_OP
DEFINE_ACTIVATION_INFERSHAPE(Tanh);
#endif  // TANH_OP

#ifdef LOG_OP
DEFINE_ACTIVATION_INFERSHAPE(Log);
#endif  // LOG_OP

#ifdef LEAKY_RELU_OP
DEFINE_ACTIVATION_INFERSHAPE(LeakyRelu);
#endif  // LEAKY_RELU_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef RELU_OP
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(relu, ops::ReluOp);
REGISTER_OPERATOR_CPU(relu6, ops::Relu6Op);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(relu, ops::ReluOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(relu, ops::ReluOp);
REGISTER_OPERATOR_CL(relu6, ops::Relu6Op);
#endif
#endif  // RELU_OP

#ifdef SIGMOID_OP
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(sigmoid, ops::SigmoidOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(sigmoid, ops::SigmoidOp);
#endif
#endif  // SIGMOID_OP

#ifdef TANH_OP
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(tanh, ops::TanhOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(tanh, ops::TanhOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(tanh, ops::TanhOp);
#endif
#endif  // TANH_OP

#ifdef PADDLE_MOBILE_CPU
#ifdef LOG_OP
REGISTER_OPERATOR_CPU(log, ops::LogOp);
#endif  // LOG_OP
#endif

#ifdef LEAKY_RELU_OP
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(leaky_relu, ops::LeakyReluOp);
#endif  // LEAKY_RELU_OP

#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(leaky_relu, ops::LeakyReluOp);
#endif
#endif
