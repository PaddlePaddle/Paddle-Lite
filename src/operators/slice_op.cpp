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

#ifdef SLICE_OP

#include "operators/slice_op.h"
#include <vector>
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void SliceOp<Dtype, T>::InferShape() const {
  auto axes = this->param_.axes_;
  auto input = this->param_.input_;
  auto output = this->param_.output_;
  PADDLE_MOBILE_ENFORCE(axes.size() == 1,
                 "axes size should equals 1");
  PADDLE_MOBILE_ENFORCE(input->dims().size() == output->dims().size(),
                        "input dim size should equals output dim size");
  PADDLE_MOBILE_ENFORCE(input->dims().size() - axes[0] == 3,
                        "op only support slice channel now");
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(slice, ops::SliceOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(slice, ops::SliceOp);
#endif
#endif
