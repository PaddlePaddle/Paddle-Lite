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

#ifdef INSTANCENORM_OP

#include "operators/instancenorm_op.h"
#include "framework/op_proto_maker.h"
#include "framework/op_registry.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void InstanceNormOp<Dtype, T>::InferShape() const {
  auto x_dims = this->param_.InputX()->dims();
  this->param_.Out()->Resize(x_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(instance_norm, ops::InstanceNormOp);
#endif

#endif
