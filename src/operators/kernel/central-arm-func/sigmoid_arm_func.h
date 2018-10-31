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
#ifdef SIGMOID_OP
#pragma once

#include <cmath>

#include "operators/op_param.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#include "operators/math/math_func_neon.h"
#endif

namespace paddle_mobile {
namespace operators {

using framework::DDim;

void sigmoid(const Tensor *X, Tensor *Y) {
#ifdef __ARM_NEON
  const float *input = X->data<float>();
  float *output = Y->mutable_data<float>();
  const DDim &dDim = X->dims();
  int axis_index = 1;
  if (dDim.size() < 4) {
    axis_index = 0;
  }
  DDim outer_ddim =
      paddle_mobile::framework::slice_ddim(dDim, 0, axis_index + 1);
  DDim inner_ddim =
      paddle_mobile::framework::slice_ddim(dDim, axis_index + 1, dDim.size());
  int out_size = paddle_mobile::framework::product(outer_ddim);
  int inner_size = paddle_mobile::framework::product(inner_ddim);

  DLOG << "outsize=" << out_size;
  DLOG << "innersize=" << inner_size;
  #pragma omp parallel for
  for (int i = 0; i < out_size; ++i) {
    const float *input_outer_ptr = input + i * inner_size;
    float *output_outer_ptr = output + i * inner_size;
    int nn = inner_size >> 2;
    int remain = inner_size - (nn << 2);
    float32x4_t _one = vdupq_n_f32(1.f);
    for (; nn > 0; nn--) {
      float32x4_t data = vld1q_f32(input_outer_ptr);
      data = vnegq_f32(data);
      data = exp_ps(data);
      data = vaddq_f32(data, _one);
      float32x4_t out_data = vrecpeq_f32(data);
      out_data = vmulq_f32(vrecpsq_f32(data, out_data), out_data);
      vst1q_f32(output_outer_ptr, out_data);

      input_outer_ptr += 4;
      output_outer_ptr += 4;
    }
    for (; remain > 0; remain--) {
      *output_outer_ptr = 1.f / (1.f + exp(-*input_outer_ptr));
      output_outer_ptr++;
      input_outer_ptr++;
    }
  }
#else
#endif
}

template <typename P>
void SigmoidCompute(const SigmoidParam<CPU> &param) {
  const Tensor *in_x = param.InputX();
  Tensor *out = param.Out();
  auto x_dims = in_x->dims();
  out->Resize(x_dims);
  sigmoid(in_x, out);
}
}  // namespace operators
}  // namespace paddle_mobile
#endif
