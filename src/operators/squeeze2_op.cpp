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

#ifdef SQUEEZE2_OP

#include "operators/squeeze2_op.h"
#include <vector>
#include "operators/kernel/squeeze2_kernel.h"
namespace paddle_mobile {
namespace operators {

static DDim GetOutputShape(const std::vector<int> &squeeze_dims,
                           const DDim &in_dims,
                           bool is_runtime) {
  size_t num_squeeze_dims = squeeze_dims.size();
  int cnt_squeezed_dims = 0;
  bool should_squeeze[9] = {false};

  // Determines number of dimensions of output tensor after squeeze.
  // Mark and count the dimensions need to be squeezed
  if (num_squeeze_dims == 0) {
    for (int idx = 0; idx < in_dims.size(); ++idx) {
      if (in_dims[idx] == 1) {
        should_squeeze[idx] = true;
        ++cnt_squeezed_dims;
      }
    }
  } else {
    for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
      int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + in_dims.size()
                                          : squeeze_dims[idx];
      // Check current index, the upper limit has been checked.
      CHECK_GE(current, 0)
          << "Invalid axis, the negative axis is out of range.";

      if (is_runtime) {
        CHECK_EQ(in_dims[current], 1) << "Invalid axis index, the axis that "
                                         "will be squeezed should be equal "
                                         "to 1.";
      }

      if (!(should_squeeze[current])) {
        ++cnt_squeezed_dims;
      }
      should_squeeze[current] = true;
    }
  }

  // Make output dimensions
  std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
  for (int in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
    if (!should_squeeze[in_idx]) {
      output_shape[out_idx++] = in_dims[in_idx];
    }
  }
  return DDim(output_shape);
}

template <typename Dtype, typename T>
void Squeeze2Op<Dtype, T>::InferShape() const {
  // auto &shape = this->param_.Shape();
  // auto input_x_dims = this->param_.InputX()->dims();
  // auto out_dims = ValidateShape(shape, input_x_dims);
  // this->param_.Out()->Resize(out_dims);
  // std::vector<int64_t> xshape_dims(input_x_dims.size() + 1, 0);
  // for (int i = 0; i < input_x_dims.size(); ++i) {
  //   xshape_dims[i + 1] = input_x_dims[i];
  // }
  // this->param_.OutputXShape()->Resize(framework::make_ddim(xshape_dims));

  std::vector<int> squeeze_dims = param_.Axes();
  DDim in_dims = param_.InputX->dims();
  DDim out_dim = GetOutputShape(squeeze_dims, in_dims, true);
  param_.Out->Resize(out_dim);

  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1, 1);
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  param_.OutputXShape()->Resize(DDim(xshape_dims));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(squeeze2, ops::Squeeze2Op);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(squeeze2, ops::Squeeze2Op);
#endif

#endif
