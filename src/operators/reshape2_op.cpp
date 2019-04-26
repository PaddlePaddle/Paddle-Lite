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

#ifdef RESHAPE2_OP

#include "operators/reshape2_op.h"
#include <vector>
#include "operators/kernel/reshape_kernel.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void Reshape2Op<Dtype, T>::InferShape() const {
  auto &shape = this->param_.Shape();
  auto input_x_dims = this->param_.InputX()->dims();
  auto out_dims = ValidateShape(shape, input_x_dims);
  this->param_.Out()->Resize(out_dims);
  std::vector<int64_t> xshape_dims(input_x_dims.size() + 1, 0);
  for (int i = 0; i < input_x_dims.size(); ++i) {
    xshape_dims[i + 1] = input_x_dims[i];
  }
  this->param_.OutputXShape()->Resize(framework::make_ddim(xshape_dims));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(reshape2, ops::Reshape2Op);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(reshape2, ops::Reshape2Op);
#endif

#endif
