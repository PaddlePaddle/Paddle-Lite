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

#ifdef RESIZE_OP

#include "operators/kernel/resize_kernel.h"
#include <cmath>

namespace paddle_mobile {
namespace operators {
void BiLinearResizeTensor(const float* src, const int src_height,
                          const int src_width, float* dst, const int dst_height,
                          const int dst_width) {
  const float scale_w = src_width / static_cast<float>(dst_width);
  const float scale_h = src_height / static_cast<float>(dst_height);
  float* dst_data = dst;
  const float* src_data = src;

  for (int dst_h = 0; dst_h < dst_height; ++dst_h) {
    float fh = dst_h * scale_h;

    int src_h = std::floor(fh);

    fh -= src_h;
    const float w_h0 = fabs(1.0 - fh);
    const float w_h1 = fabs(fh);

    const int dst_offset_1 = dst_h * dst_width;
    const int src_offset_1 = src_h * src_width;

    float* dst_data_ptr = dst_data + dst_offset_1;

    for (int dst_w = 0; dst_w < dst_width; ++dst_w) {
      float fw = dst_w * scale_w;
      int src_w = std::floor(fw);
      fw -= src_w;
      const float w_w0 = fabs(1.0 - fw);
      const float w_w1 = fabs(fw);

      float dst_value = 0;

      const int src_idx = src_offset_1 + src_w;
      dst_value += (w_h0 * w_w0 * src_data[src_idx]);
      int flag = 0;
      if (src_w + 1 < src_width) {
        dst_value += (w_h0 * w_w1 * src_data[src_idx + 1]);
        ++flag;
      }
      if (src_h + 1 < src_height) {
        dst_value += (w_h1 * w_w0 * src_data[src_idx + src_width]);
        ++flag;
      }

      if (flag > 1) {
        dst_value += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
        //                ++flag;
      }
      *(dst_data_ptr++) = dst_value;
    }
  }
}

void ResizeTensor(const Tensor* src, const int src_n, const int src_c,
                  Tensor* dst, const int dst_n, const int dst_c) {
  framework::DDim in_dims = src->dims();
  const int src_chans = in_dims[1];
  const int src_height = in_dims[2];
  const int src_width = in_dims[3];
  const int src_offset = (src_n * src_chans + src_c) * src_height * src_width;

  framework::DDim out_dims = dst->dims();
  const int dst_chans = out_dims[1];
  const int dst_height = out_dims[2];
  const int dst_width = out_dims[3];
  const int dst_offset = (dst_n * dst_chans + dst_c) * dst_height * dst_width;

  const auto* src_ptr = src->data<float>();
  auto* dst_ptr = dst->data<float>();
  const auto* src_data = &(src_ptr[src_offset]);
  auto* dst_data = &(dst_ptr[dst_offset]);
  BiLinearResizeTensor(src_data, src_height, src_width, dst_data, dst_height,
                       dst_width);
}

void ResizeTensor(const Tensor* src, Tensor* dst) {
  framework::DDim in_dims = src->dims();
  framework::DDim out_dims = dst->dims();
  PADDLE_MOBILE_ENFORCE(in_dims[0] == out_dims[0],
                        "src tensor batch num not equal to dst tensor");
  PADDLE_MOBILE_ENFORCE(in_dims[1] == out_dims[1],
                        "src tensor channel num not equal to dst tensor");
  for (int n = 0, batch_num = in_dims[0]; n < batch_num; ++n) {
    for (int c = 0, chan_num = in_dims[1]; c < chan_num; ++c) {
      ResizeTensor(src, n, c, dst, n, c);
    }
  }
}

template <>
void ResizeKernel<CPU, float>::Compute(const ResizeParam<CPU>& param) {
  const auto* input_x = param.InputX();
  const auto& input_x_dims = input_x->dims();
  auto* out = param.Out();
  framework::DDim out_dims = CalOutputShape(param);

  out->Resize(out_dims);
  ResizeTensor(input_x, out);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
