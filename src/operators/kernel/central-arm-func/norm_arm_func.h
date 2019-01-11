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

#ifdef NORM_OP

#pragma once

#include <cmath>
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

inline void GetDims(const framework::DDim &dim, int axis, int *pre, int *n,
                    int *post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

template <typename P>
void NormCompute(const NormParam<CPU> &param) {
  const float epsilon = param.Epsilon();
  int axis = param.Axis();

  const framework::Tensor *input = param.InputX();
  framework::Tensor square;
  framework::Tensor *norm = param.OutputNorm();
  framework::Tensor *out = param.Out();

  auto x_dims = input->dims();
  if (axis < 0) {
    axis += x_dims.size();
  }

  int pre, n, post;
  GetDims(x_dims, axis, &pre, &n, &post);

  framework::DDim shape = {pre, n, post};
  framework::DDim norm_shape = {pre, post};
  square.Resize(shape);

  const float *input_ptr = input->data<float>();
  float *square_ptr = square.mutable_data<float>();
  float *norm_ptr = norm->mutable_data<float>();
  float *out_ptr = out->mutable_data<float>();

  const float *in_tmp = input_ptr;
  float *square_tmp = square_ptr;
  for (int i = 0; i < input->numel(); ++i) {
    float element = *in_tmp;
    *square_tmp = element * element;
    square_tmp++;
    in_tmp++;
  }

  //  const float *norm_tmp = norm_ptr;
  //  for (int i = 0; i < norm->numel(); ++i) {
  //    *norm_tmp = 0;
  //    norm_tmp++;
  //  }

  square_tmp = square_ptr;
  float *norm_tmp = norm_ptr;
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < post; ++j) {
      for (int k = 0; k < n; ++k) {
        if (k == 0) {
          *norm_tmp = *square_tmp;
        } else {
          *norm_tmp += *(square_tmp + k * post);
        }
      }
      float sum = *norm_tmp + epsilon;
      *norm_tmp = sqrtf(sum);
      norm_tmp++;
      square_tmp++;
    }
  }

  in_tmp = input_ptr;
  norm_tmp = norm_ptr;
  float *out_tmp = out_ptr;
  for (int i = 0; i < pre; ++i) {
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < post; ++j) {
        *out_tmp = *in_tmp / *norm_tmp;
        in_tmp++;
        norm_tmp++;
        out_tmp++;
      }
      out_tmp = out_ptr + i * post;
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
