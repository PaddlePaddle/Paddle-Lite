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

#ifdef POOL_OP

#include "operators/pool_op.h"
#include <vector>
#include "framework/op_proto_maker.h"
#include "framework/op_registry.h"

namespace paddle_mobile {
namespace operators {

int PoolOutputSize(int input_size, int filter_size, int padding, int stride,
                   bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}
template <typename DeviceType, typename T>
void PoolOp<DeviceType, T>::InferShape() const {
  auto in_x_dims = this->param_.Input()->dims();
  std::vector<int> ksize = this->param_.Ksize();
  std::vector<int> paddings = this->param_.Paddings();
  std::vector<int> strides = this->param_.Strides();
  bool ceil_mode = this->param_.isCeilMode();

  if (this->param_.isGlobalPooling()) {
    ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(in_x_dims[i + 2]);
    }
  }
  std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
  for (size_t i = 0; i < ksize.size(); ++i) {
    output_shape.push_back(PoolOutputSize(in_x_dims[i + 2], ksize[i],
                                          paddings[i], strides[i], ceil_mode));
  }
  this->param_.Output()->Resize(framework::make_ddim(output_shape));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(pool2d, ops::PoolOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(pool2d, ops::PoolOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(pool2d, ops::PoolOp);
#endif

#endif
