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

#include "operators/fusion_dequant_bn_op.h"

namespace paddle_mobile {
namespace operators {

#ifdef FUSION_DEQUANT_BN_OP
template <typename Dtype, typename T>
void FusionDequantBNOp<Dtype, T>::InferShape() const {
  const auto& input_dims = this->param_.input_->dims();
  this->param_.output_->Resize(input_dims);
}
#endif  // FUSION_DEQUANT_BN_OP

#ifdef FUSION_DEQUANT_BN_RELU_OP
template <typename Dtype, typename T>
void FusionDequantBNReluOp<Dtype, T>::InferShape() const {
  const auto& input_dims = this->param_.input_->dims();
  this->param_.output_->Resize(input_dims);
}
#endif  // FUSION_DEQUANT_BN_RELU_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef FUSION_DEQUANT_BN_OP
REGISTER_FUSION_MATCHER(fusion_dequant_bn, ops::FusionDequantBNMatcher);
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(fusion_dequant_bn, ops::FusionDequantBNOp);
#endif  // PADDLE_MOBILE_CPU
#endif  // FUSION_DEQUANT_BN_OP

#ifdef FUSION_DEQUANT_BN_RELU_OP
REGISTER_FUSION_MATCHER(fusion_dequant_bn_relu,
                        ops::FusionDequantBNReluMatcher);
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(fusion_dequant_bn_relu, ops::FusionDequantBNReluOp);
#endif  // PADDLE_MOBILE_CPU
#endif  // FUSION_DEQUANT_BN_RELU_OP
