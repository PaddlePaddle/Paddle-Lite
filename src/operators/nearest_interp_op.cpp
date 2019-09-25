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

#ifdef NEAREST_INTERP_OP

#include "operators/nearest_interp_op.h"
#include <vector>
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void NearestInterpOp<Dtype, T>::InferShape() const {
  auto x_dims = this->param_.InputX()->dims();
  framework::DDim out_dims(x_dims);
 
  int outHeight = this->param_.OutHeight();
  int outWidth = this->param_.OutWidth();

  if (outHeight != 0 && outWidth != 0) {
  	out_dims[2] = this->param_.OutHeight();
  	out_dims[3] = this->param_.OutWidth();
  } else {
  	int scale = static_cast<int>(this->param_.Scale());
  	out_dims[2] = x_dims[2] * scale;
  	out_dims[3] = x_dims[3] * scale;
  }
  this->param_.Out()->Resize(out_dims);

}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(nearest_interp, ops::NearestInterpOp);
#endif

#endif
