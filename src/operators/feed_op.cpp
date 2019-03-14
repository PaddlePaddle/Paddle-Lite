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

#include "operators/feed_op.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void FeedOp<DeviceType, T>::InferShape() const {
  auto out_dims = this->param_.Out()->dims();
  out_dims[0] = this->param_.BatchSize();
  int col = this->param_.Col();
  auto input_dims = this->param_.InputX()->at(col).dims();
  if (input_dims.size() == 4) {
    this->param_.Out()->Resize(input_dims);
  } else {
    this->param_.Out()->Resize(out_dims);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(feed, ops::FeedOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(feed, ops::FeedOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(feed, ops::FeedOp);
#endif
