/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "framework/operator.h"
#include "operators/op_param.h"
#pragma once;

namespace paddle_mobile {
namespace operators {

using namespace framework;

template <typename T> struct LRNFunctor {
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
            for (int c = 0; c < H; c++) {
              for (int d = 0; d < W; d++) {
                int u = a * stride0 + b * stride1 + c * stride2 + d;

                int i = a * stride0 + channel * stride1 + c * stride2 + d;

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
} // namespace operators
} // namespace paddle_mobile
