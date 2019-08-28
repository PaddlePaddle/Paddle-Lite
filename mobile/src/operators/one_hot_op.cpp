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

#ifdef ONE_HOT_OP

#include "operators/one_hot_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void OnehotOp<Dtype, T>::InferShape() const {
  const auto &x_dims = this->param_.input_->dims();
  int depth = this->param_.depth_;
  framework::DDim out_dims(x_dims);
  out_dims[out_dims.size() - 1] = depth;
  this->param_.output_->Resize(out_dims);
  if (std::is_same<DeviceType<kCPU>, Dtype>::value) {
    this->param_.output_->set_lod(this->param_.input_->lod());
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(one_hot, ops::OnehotOp);
#endif

#endif  // ONE_HOT_OP
