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

#ifdef LRN_OP

#pragma once

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

using namespace framework;

template <typename T>
struct LRNFunctor {
  void operator()(const framework::Tensor &input, framework::Tensor *out, int N,
                  int C, int H, int W, int n, T k, T alpha, T beta) {
    auto input_ptr = input.data<T>();
    const int start = -(n - 1) / 2;
    const int end = start + n;

    const int stride0 = C * H * W;
    const int stride1 = H * W;
    const int stride2 = W;
    const int stride3 = 1;

    framework::Tensor sqr_buffer;
    auto sqr_buffer_ptr = sqr_buffer.mutable_data<T>(input.dims());
    std::fill(sqr_buffer_ptr, sqr_buffer_ptr + sqr_buffer.numel(), k);
    for (int a = 0; a < N; a++) {
      for (int b = 0; b < C; b++) {
        for (int index = start; index < end; index++) {
          int channel = b + index;
          if (channel >= 0 && channel < C) {
            int tmp_u = a * stride0 + b * stride1;
            int tmp_i = a * stride0 + channel * stride1;
            for (int c = 0; c < H; c++) {
              for (int d = 0; d < W; d++) {
                int tmp = c * stride2 + d;
                int u = tmp_u + tmp;
                int i = tmp_i + tmp;
                sqr_buffer_ptr[u] += alpha * input_ptr[i] * input_ptr[i];
              }
            }
          }
        }
      }
    }
    auto out_ptr = out->data<T>();
    for (int i = 0; i < input.numel(); i++) {
      out_ptr[i] = input_ptr[i] / pow(sqr_buffer_ptr[i], beta);
    }
  }
};

template <typename DeviceType, typename T>
class LrnKernel : public framework::OpKernelBase<DeviceType, LrnParam> {
 public:
  void Compute(const LrnParam &param) const;
};
}  // namespace operators
}  // namespace paddle_mobile

#endif
