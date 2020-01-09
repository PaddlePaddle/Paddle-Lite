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

#ifdef EXPAND_OP

#include "operators/expand_op.h"
#include <framework/ddim.h>

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ExpandOp<Dtype, T>::InferShape() const {
  auto x_dim = this->param_.InputX()->dims();

  int expand_size = this->param_.expand_times.size();
  int x_dims_size = x_dim.size();
  PADDLE_MOBILE_ENFORCE(expand_size == x_dims_size,
                        "The number of expand_times size must be qual to the "
                        "rank of Input(X). The number of expand_times size "
                        "must be qual to the rank of Input(X).")

  framework::DDim out_dims(this->param_.InputX()->dims());
  for (size_t i = 0; i < this->param_.expand_times.size(); ++i) {
    out_dims[i] *= this->param_.expand_times[i];
  }
  this->param_.Out()->Resize(out_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(expand, ops::ExpandOp);
#endif

#endif
