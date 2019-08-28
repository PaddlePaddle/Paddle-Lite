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

#ifdef LOD_RESET_OP

#include "operators/lod_reset_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void LodResetOp<Dtype, T>::InferShape() const {
  const auto &input_dims = this->param_.input_x_->dims();
  this->param_.output_->Resize(input_dims);
  if (std::is_same<DeviceType<kCPU>, Dtype>::value) {
    if (this->param_.append) {
      this->param_.output_->set_lod(this->param_.input_x_->lod());
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(lod_reset, ops::LodResetOp);
#endif

#endif  // LOD_RESET_OP
