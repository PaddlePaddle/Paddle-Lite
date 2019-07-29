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
#ifdef PADDLE_MOBILE_CL
  auto output_dims = output->dims();
  auto output_dims_size = output_dims.size();
  bool should_resize = true;
  if (output_dims_size > 4) {
    for (int i = 0; i < output_dims_size - 4; ++i) {
      if (output_dims[i] != 0 && output_dims[i] != 1) {
        should_resize = false;
        break;
      }
    }
    if (should_resize) {
      std::vector<int64_t> temp_output_dims;
      temp_output_dims.reserve(static_cast<size_t>(4));
      for (int i = output_dims_size - 4; i < output_dims_size; ++i) {
        temp_output_dims.push_back(output_dims[i]);
      }
      framework::DDim temp_ddim = framework::make_ddim(temp_output_dims);
      this->param_.output_->Resize(temp_ddim);
    }
  }
#endif
  PADDLE_MOBILE_ENFORCE(axes.size() == 1, "axes size should equals 1");
  PADDLE_MOBILE_ENFORCE(input->dims().size() == output->dims().size(),
                        "input dim size should equals output dim size");
#ifdef PADDLE_MOBILE_CL
  PADDLE_MOBILE_ENFORCE(
      input->dims().size() -
              (axes[0] - (this->param_.original_output_dims_size_ -
                          this->param_.output_->dims().size())) ==
          3,
      "op only support slice channel now");
#else
  PADDLE_MOBILE_ENFORCE(input->dims().size() - axes[0] == 3,
                        "op only support slice channel now");
#endif
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
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(slice, ops::SliceOp);
#endif
#endif  // SLICE_OP
