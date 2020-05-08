/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void reduce_prod_n(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = channel_in * hw_size;
  int data_index, src_index, src_index0;
  for (int c = 0; c < channel_in; ++c) {
    for (int h = 0; h < height_in; ++h) {
      for (int w = 0; w < width_in; ++w) {
        data_index = c * hw_size + h * width_in + w;
        dst[data_index] = static_cast<T>(1);
        for (int n = 0; n < num_in; ++n) {
          src_index = n * chw_size + data_index;
          dst[data_index] *= src[src_index];
        }
      }
    }
  }
}

template <typename T>
void reduce_prod_c(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = hw_size * channel_in;
  int data_index, src_index0, src_index;
  for (int n = 0; n < num_in; ++n) {
    for (int h = 0; h < height_in; ++h) {
      for (int w = 0; w < width_in; ++w) {
        data_index = n * hw_size + h * width_in + w;
        src_index0 = n * chw_size + h * width_in + w;
        dst[data_index] = static_cast<T>(1);
        for (int c = 0; c < channel_in; ++c) {
          src_index = src_index0 + c * hw_size;
          dst[data_index] *= src[src_index];
        }
      }
    }
  }
}

template <typename T>
void reduce_prod_h(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int cw_size = channel_in * width_in;
  int chw_size = cw_size * height_in;
  int hw_size = height_in * width_in;
  int data_index, src_index, src_index0;
  for (int n = 0; n < num_in; ++n) {
    for (int c = 0; c < channel_in; ++c) {
      for (int w = 0; w < width_in; ++w) {
        data_index = n * cw_size + c * width_in + w;
        src_index0 = n * chw_size + c * hw_size + w;
        dst[data_index] = static_cast<T>(1);
        for (int h = 0; h < height_in; ++h) {
          src_index = src_index0 + h * width_in;
          dst[data_index] *= src[src_index];
        }
      }
    }
  }
}

template <typename T>
void reduce_prod_w(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int ch_size = channel_in * height_in;
  int hw_size = height_in * width_in;
  int chw_size = ch_size * width_in;
  int data_index = 0;
  int src_index0 = 0;
  int src_index = 0;
  for (int n = 0; n < num_in; ++n) {
    for (int c = 0; c < channel_in; ++c) {
      for (int h = 0; h < height_in; ++h) {
        data_index = n * ch_size + c * height_in + h;
        src_index0 = n * chw_size + c * hw_size + h * width_in;
        dst[data_index] = static_cast<T>(1);
        for (int w = 0; w < width_in; ++w) {
          src_index = src_index0 + w;
          dst[data_index] *= src[src_index];
        }
      }
    }
  }
}

template <typename T>
void reduce_prod_nc(const T* src,
                    T* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  // reduce n first.
  DDimLite ddimA({1, channel_in, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  auto* tmp_out = tensor_tmp.mutable_data<T>();
  reduce_prod_n(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_prod_c(tmp_out, dst, 1, channel_in, height_in, width_in);
}

template <typename T>
void reduce_prod_ch(const T* src,
                    T* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  // reduce c first
  DDimLite ddimA({num_in, 1, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  auto* tmp_out = tensor_tmp.mutable_data<T>();
  reduce_prod_c(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_prod_h(tmp_out, dst, num_in, 1, height_in, width_in);
}

template <typename T>
void reduce_prod_hw(const T* src,
                    T* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  // reduce h first
  DDimLite ddimA({num_in, channel_in, 1, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  auto* tmp_out = tensor_tmp.mutable_data<T>();
  reduce_prod_h(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_prod_w(tmp_out, dst, num_in, channel_in, 1, width_in);
}

template <typename T>
void reduce_prod_all(const T* src, T* dst, int64_t total_num) {
  dst[0] = static_cast<T>(1);
  for (int n = 0; n < total_num; ++n) {
    dst[0] *= src[n];
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
