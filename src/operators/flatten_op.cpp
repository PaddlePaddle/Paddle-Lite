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

#ifdef FLATTEN_OP

#include "operators/flatten_op.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void FlattenOp<DeviceType, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.InputX() != nullptr,
                        "Input (X) of Flatten op should not be null.");
  PADDLE_MOBILE_ENFORCE(this->param_.Out() != nullptr,
                        "Output (Output) of Flatten op should not be null.");

  auto &axis = this->param_.Axis();
  PADDLE_MOBILE_ENFORCE(axis >= 0,
                        "The axis should be greater than or equal to 0.");

  auto &in_dims = this->param_.InputX()->dims();
  PADDLE_MOBILE_ENFORCE(
      axis <= in_dims.size(),
      "The axis should be less than or equal to input tensor's rank.");

  const auto &out_dims = GetOutputShape(axis, in_dims);
  // this->param_.Out()->Resize(in_dims);
  this->param_.Out()->Resize(framework::make_ddim(out_dims));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(flatten, ops::FlattenOp);
REGISTER_OPERATOR_CPU(flatten2, ops::Flatten2Op);
#endif

#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(flatten, ops::FlattenOp);
REGISTER_OPERATOR_FPGA(flatten2, ops::Flatten2Op);
#endif

#endif  // FLATTEN_OP
