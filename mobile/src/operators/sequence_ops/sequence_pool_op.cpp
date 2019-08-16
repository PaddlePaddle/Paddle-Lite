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

#ifdef SEQUENCE_POOL_OP

#include "operators/sequence_ops/sequence_pool_op.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void SequencePoolOp<DeviceType, T>::InferShape() const {
  const auto *input = this->param_.input_;
  auto out_dims = input->dims();
  out_dims[0] = input->lod()[0].size() - 1;
  this->param_.output_->Resize(out_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(sequence_pool, ops::SequencePoolOp);
#endif

#endif  // SEQUENCE_POOL_OP
