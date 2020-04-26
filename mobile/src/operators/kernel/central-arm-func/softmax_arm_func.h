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

#ifdef SOFTMAX_OP
#pragma once
#include "../../math/softmax.h"
#include "operators/op_param.h"
namespace paddle_mobile {
namespace operators {

void softmax_basic_axis_float(const float *din, float *dout,
                              const int axis_size, const int inner_num,
                              const int outer_num) {
  int compute_size = inner_num * outer_num;
#pragma omp parallel for
  for (int i = 0; i < compute_size; ++i) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    float max_data = din[real_index];
    // get max
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      max_data = din[real_index] > max_data ? din[real_index] : max_data;
    }

    real_index = idx_outer * inner_num + idx_inner;
    // sub, exp and sum
    dout[real_index] = expf(din[real_index] - max_data);
    float sum_data = dout[real_index];
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      dout[real_index] = expf(din[real_index] - max_data);
      sum_data += dout[real_index];
    }

    float sum_inv = 1.f / sum_data;
    real_index = idx_outer * inner_num + idx_inner;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      dout[real_index] *= sum_inv;
      real_index += inner_num;
    }
  }
}

template <typename P>
void SoftmaxCompute(const SoftmaxParam<CPU> &param) {
  const Tensor *in_x = param.InputX();
  Tensor *out = param.Out();
  auto x_dims = in_x->dims();
  out->Resize(x_dims);
  out->mutable_data<float>();
  if (param.has_axis_) {
    int axis = param.axis_;
    int axis_size = x_dims[axis];
    auto x_rank = x_dims.size();
    DLOG << "x_rank :" << x_rank;

    if (axis < 0) {
      axis += x_rank;
    }

    DLOG << "axis :" << axis;

    int outer_num = framework::product(framework::slice_ddim(x_dims, 0, axis));
    DLOG << "outer_num :" << outer_num;
    int inner_num =
        framework::product(framework::slice_ddim(x_dims, axis + 1, x_rank));
    DLOG << "inner_num :" << inner_num;

    softmax_basic_axis_float(in_x->data<float>(), out->data<float>(), axis_size,
                             inner_num, outer_num);
  } else {
    math::SoftmaxFuntor<CPU, float>()(in_x, out);
  }
}
}  // namespace operators
}  // namespace paddle_mobile
#endif
