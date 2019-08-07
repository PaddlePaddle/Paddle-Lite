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

#include "operators/kernel/slice_kernel.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype>
void SliceCompute(const SliceParam<CPU>& param) {
  auto input = param.input_;
  auto output = param.output_;
  auto* input_ptr = input->data<float>();
  auto* output_ptr = output->mutable_data<float>();
  auto out_dims = output->dims();
  auto in_dims = input->dims();
  auto starts = param.starts_;
  auto ends = param.ends_;
  int axes = param.axes_[0];
  int HW = input->dims()[axes + 1] * input->dims()[axes + 2];
  int batch_size = out_dims[axes - 1];
  int input_channel = in_dims[axes];
  int output_channel = out_dims[axes];

  for (int c1 = 0; c1 < batch_size; ++c1) {
    for (int c2 = starts[0], c3 = 0; c2 < ends[0]; ++c2, ++c3) {
      size_t out_offset = c1 * output_channel * HW + c3 * HW;
      size_t in_offset = c1 * input_channel * HW + c2 * HW;
      memcpy(output_ptr + out_offset, input_ptr + in_offset,
             HW * sizeof(float));
    }
  }
}

template <>
bool SliceKernel<CPU, float>::Init(SliceParam<CPU>* param) {
  return true;
}

template <>
void SliceKernel<CPU, float>::Compute(const SliceParam<CPU>& param) {
  int rank = param.input_->dims().size();
  switch (rank) {
    case 1:
      if (param.input_->type() == type_id<int>().hash_code()) {
        SliceCompute<int>(param);
      } else if (param.input_->type() == type_id<float>().hash_code()) {
        SliceCompute<float>(param);
      }
      break;
    case 2:
      SliceCompute<float>(param);
      break;
    case 4:
      SliceCompute(param);
      break;
    case 5:
      if (param.input_->dims()[0] == 1) {
        SliceCompute(param);
      }
      break;
    default:
      PADDLE_MOBILE_ENFORCE(0, "input dims not support now");
      break;
  }
}

}  // namespace operators
}  // namespace paddle_mobile
#endif
