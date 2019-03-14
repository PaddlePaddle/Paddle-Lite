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

#ifdef SHAPE_OP

#include "operators/shape_op.h"

namespace paddle_mobile {
namespace operators {
template <typename DeviceType, typename T>
void ShapeOp<DeviceType, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.Input() != nullptr,
                        "Input (Input) of get_shape op should not be null.");
  PADDLE_MOBILE_ENFORCE(this->param_.Out() != nullptr,
                        "Output (Out) of get_shape op should not be null.");
  this->param_.Out()->Resize({this->param_.Input()->dims().size()});
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(shape, ops::ShapeOp);
#endif

#endif
