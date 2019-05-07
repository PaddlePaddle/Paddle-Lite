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

#ifdef TRANSPOSE_OP

#include <vector>

#include "common/enforce.h"
#include "operators/transpose_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void TransposeOp<Dtype, T>::InferShape() const {
  auto input_x_dims = this->param_.InputX()->dims();
  auto axis = this->param_.Axis();

  size_t x_dims_size = input_x_dims.size();
  size_t axis_size = axis.size();

  PADDLE_MOBILE_ENFORCE((x_dims_size == axis_size),
                        "input_dims must "
                        "be equal to the axis_size. ")

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    PADDLE_MOBILE_ENFORCE(
        axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
        "Each element of Attribute axis should be a unique value "
        "range from 0 to (dims - 1), "
        "where the dims is the axis's size");
  }
  framework::DDim out_dims(input_x_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = input_x_dims[axis[i]];
  }
  this->param_.Out()->Resize(out_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(transpose, ops::TransposeOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(transpose, ops::TransposeOp);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(transpose, ops::TransposeOp);
#endif

#endif  // TRANSPOSE_OP
