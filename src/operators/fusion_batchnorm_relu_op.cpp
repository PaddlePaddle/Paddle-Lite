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

#ifdef FUSION_BN_RELU_OP

#include "operators/fusion_batchnorm_relu_op.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void FusionBatchnormReluOp<Dtype, T>::InferShape() const {
  auto x_dims = this->param_.InputX()->dims();
  this->param_.OutputY()->Resize(x_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

// namespace ops = paddle_mobile::operators;
// REGISTER_FUSION_MATCHER(fusion_batchnorm_relu,
// ops::FusionBatchnormReluMatcher);

// #if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
// REGISTER_OPERATOR_FPGA(fusion_batchnorm_relu, ops::FusionBatchnormReluOp);
// #endif

#endif
