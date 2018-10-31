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

#ifdef POLYGONBOXTRANSFORM_OP

#include "operators/polygon_box_transform_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void PolygonBoxTransformOp<Dtype, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.Input() != nullptr,
                        "Input (Input) of get_shape op should not be null.");
  PADDLE_MOBILE_ENFORCE(this->param_.Output() != nullptr,
                        "Output (Output) of get_shape op should not be null.");

  auto input_dims = this->param_.Input()->dims();

  PADDLE_MOBILE_ENFORCE(input_dims.size() == 4, "input's rank must be 4.");
  PADDLE_MOBILE_ENFORCE(input_dims[1] % 2 == 0,
                        "input's second dimension must be even.");

  this->param_.Output()->Resize(input_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(polygon_box_transform, ops::PolygonBoxTransformOp);
#endif

#endif
