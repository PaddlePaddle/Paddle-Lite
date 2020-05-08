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

#ifdef PIXEL_SHUFFLE_OP

#include "operators/pixel_shuffle_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void PixelShuffleOp<Dtype, T>::InferShape() const {
  auto x_dims = this->param_.InputX()->dims();
  int n = x_dims[0];
  int c = x_dims[1];
  int h = x_dims[2];
  int w = x_dims[3];
  int upscale_factor = this->param_.upscale_factor();
  this->param_.Out()->Resize(
      framework::make_ddim({n, c / (upscale_factor * upscale_factor),
                            h * upscale_factor, w * upscale_factor}));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(pixel_shuffle, ops::PixelShuffleOp);
#endif

#endif
