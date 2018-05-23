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

#pragma once

#include "operators/kernel/batchnorm_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
void BatchNormKernel<CPU, float>::Compute(const BatchNormParam &param) const {
  /// todo: test.
  const Tensor *input_x = param.InputX();
  auto input_x_ptr = input_x->data<float>();
  const auto &x_dims = input_x->dims();
  const int N = x_dims[0];
  const int C = x_dims[1];
  const int H = x_dims[2];
  const int W = x_dims[3];
  const int stride0 = C * H * W;
  const int stride1 = H * W;
  const int stride2 = W;
  Tensor *out = param.OutputY();
  auto out_ptr = out->mutable_data<float>();
  const float epsilon = param.Epsilon();
  const Tensor *mean = param.InputMean();
  const Tensor *variance = param.InputVariance();
  const Tensor *scale = param.InputScale();
  const Tensor *bias = param.InputBias();
  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  Tensor inv_std;
  auto inv_std_ptr = inv_std.mutable_data<float>(make_ddim({C}));
  if (C != variance->numel()) {
    std::cout << "C must equal to variance.numel()" << std::endl;
  }
  assert(C == variance->numel());

  /// std = (var + epsilon).sqrt();
  /// inv_std = 1 / std;
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }

  Tensor new_scale;
  auto new_scale_ptr = new_scale.mutable_data<float>(make_ddim({C}));
  Tensor new_bias;
  auto new_bias_ptr = new_bias.mutable_data<float>(make_ddim({C}));

  /// ((x - est_mean) * (inv_var) * scale + bias equal to
  /// (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
    {
      for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
          for (int w = 0; w < W; w++) {
            int index = n * stride0 + i * stride1 + h * stride2 + w;
            out_ptr[index] =
                input_x_ptr[index] * new_scale_ptr[i] + new_bias_ptr[i];
          }
        }
      }
    }
  }
  DLOG << "input[2,5,1,0](input[102]) ,channel 5 :";
  DLOG << "input_x_ptr : " << input_x_ptr[102];
  DLOG << "variance : " << variance_ptr[5];
  DLOG << "inv_std_ptr : " << inv_std_ptr[5];
  DLOG << "new_scale_ptr : " << new_scale_ptr[5];
  DLOG << "new_bias_ptr : " << new_bias_ptr[5];
  DLOG << "out_ptr : " << out_ptr[102];
}
}  // namespace operators
}  // namespace paddle_mobile
