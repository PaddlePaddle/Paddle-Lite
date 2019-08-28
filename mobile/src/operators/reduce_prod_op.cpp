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

#ifdef REDUCE_PROD_OP

#include "operators/reduce_prod_op.h"
#include <algorithm>
#include <vector>

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ReduceProdOp<Dtype, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.Input() != nullptr,
                        "Input (X) of ReduceOp op should not be null.");
  PADDLE_MOBILE_ENFORCE(this->param_.Output() != nullptr,
                        "Output (Output) of ReduceOp op should not be null.");

  auto x_dims = this->param_.Input()->dims();
  auto x_rank = x_dims.size();
  PADDLE_MOBILE_ENFORCE(x_rank <= 6,
                        "Tensors with rank at most 6 are supported.");
  auto dims = this->param_.getDim();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) dims[i] = x_rank + dims[i];
    PADDLE_MOBILE_ENFORCE(
        dims[i] < x_rank,
        "The dim should be in the range [-rank(input), rank(input)).");
  }
  sort(dims.begin(), dims.end());
  bool reduce_all = this->param_.isReduceAll();
  bool keep_dim = this->param_.isKeepDim();
  if (reduce_all) {
    if (keep_dim)
      this->param_.Output()->Resize(
          framework::make_ddim(std::vector<int64_t>(x_rank, 1)));
    else
      this->param_.Output()->Resize({1});
  } else {
    auto dims_vector = vectorize(x_dims);
    if (keep_dim) {
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = 1;
      }
    } else {
      const int kDelFlag = -2;
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = kDelFlag;
      }
      dims_vector.erase(
          remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
          dims_vector.end());
    }
    auto out_dims = framework::make_ddim(dims_vector);
    this->param_.Output()->Resize(out_dims);
    if (std::is_same<DeviceType<kCPU>, Dtype>::value) {
      if (dims[0] != 0) {
        // Only pass LoD when not reducing on the first dim.
        this->param_.Output()->set_lod(this->param_.Input()->lod());
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(reduce_prod, ops::ReduceProdOp);
#endif

#endif  // REDUCE_PROD_OP
