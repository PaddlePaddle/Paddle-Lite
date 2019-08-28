/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef FLATTEN2_OP
#include "operators/flatten2_op.h"
#include <operators/kernel/reshape_kernel.h>

namespace paddle_mobile {
namespace operators {
template <typename DeviceType, typename T>
void Flatten2Op<DeviceType, T>::InferShape() const {
  const auto* input = this->param_.InputX();
  auto* output = this->param_.Out();
  auto input_x_dims = input->dims();
  if (input->dims().size() == 4) {
    PADDLE_MOBILE_ENFORCE(this->param_.Axis() == 1,
                          "flatten 2 only support axis == 1");
    if (this->param_.Axis() == 1) {
      std::vector<int> temp_output_dims(2);
      temp_output_dims[0] = input->dims()[0];
      temp_output_dims[1] =
          input->dims()[1] * input->dims()[2] * input->dims()[3];
      output->Resize(framework::make_ddim(temp_output_dims));
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(flatten2, ops::Flatten2Op);
#endif

#endif
