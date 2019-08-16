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

#ifdef SUM_OP

#include <vector>

#include "operators/sum_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void SumOp<Dtype, T>::InferShape() const {
  auto inputs = this->param_.Inputs();
  const size_t n = inputs.size();

  std::vector<framework::DDim> inputs_dims;
  inputs_dims.reserve(n);
  for (int i = 0; i < n; i++) {
    inputs_dims.push_back(inputs[i]->dims());
  }

  if (n == 1) {
    DLOG << "Warning: sum op have only one input, "
            "may waste memory";
  }

  framework::DDim in_dim({0});

  for (auto& x_dim : inputs_dims) {
    if (framework::product(x_dim) == 0) {
      continue;
    }
    if (framework::product(in_dim) == 0) {
      in_dim = x_dim;
    } else {
      PADDLE_MOBILE_ENFORCE(in_dim == x_dim,
                            "input tensors must have same shape");
    }
  }

  this->param_.Out()->Resize(in_dim);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(sum, ops::SumOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif

#endif
