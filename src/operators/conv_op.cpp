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

#ifdef CONV_OP

#include "operators/conv_op.h"
#include <vector>
#include "framework/op_proto_maker.h"
#include "framework/op_registry.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void ConvOp<Dtype, T>::InferShape() const {
  auto in_dims = param_.Input()->dims();
  auto filter_dims = param_.Filter()->dims();
  const std::vector<int> &strides = param_.Strides();
  std::vector<int> paddings = param_.Paddings();
  int groups = param_.Groups();
  std::vector<int> dilations = param_.Dilations();

  PADDLE_MOBILE_ENFORCE((in_dims.size() == filter_dims.size() &&
                         dilations.size() == paddings.size() &&
                         paddings.size() == strides.size()),
                        "ConvParam is not suitable");

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2], filter_dims[i + 2],
                                          dilations[i], paddings[i],
                                          strides[i]));
  }

  framework::DDim ddim = framework::make_ddim(output_shape);
  param_.Output()->Resize(ddim);
}

template class ConvOp<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
USE_OP(conv2d);
REGISTER_OPERATOR(conv2d, ops::ConvOp);

#endif
