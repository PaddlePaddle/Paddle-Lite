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

#ifdef IM2SEQUENCE_OP

#include "operators/im2sequence_op.h"
#include <vector>

namespace paddle_mobile {
namespace operators {

int Im2SequenceOutputSize(int input_size, int kernel, int padding_1,
                          int padding_2, int stride) {
  int output_size =
      1 + (padding_1 + padding_2 + input_size - kernel + stride - 1) / stride;
  return output_size;
}

template <typename Dtype, typename T>
void Im2SequenceOp<Dtype, T>::InferShape() const {
  auto in_x_dims = this->param_.Input()->dims();
  const std::vector<int> &kernels = this->param_.Kernels();
  const std::vector<int> &strides = this->param_.Strides();
  std::vector<int> paddings = this->param_.Paddings();
  std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});

  for (size_t i = 0; i < strides.size(); ++i) {
    output_shape.push_back(Im2SequenceOutputSize(in_x_dims[i + 2], kernels[i],
                                                 paddings[i], paddings[i + 2],
                                                 strides[i]));
  }
  framework::DDim ddim = framework::make_ddim(output_shape);
  this->param_.Output()->Resize(ddim);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(im2sequence, ops::Im2SequenceOp);
#endif

#endif  // IM2SEQUENCE_OP
