// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <class T>
static inline T __attribute__((__always_inline__)) naive_relu(T a) {
  return a > 0 ? a : 0;
}

template <class T>
static inline T __attribute__((__always_inline__)) naive_tanh(T a) {
  float x = expf(a);
  float y = expf(-a);
  return (x - y) / (x + y);
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_add(T l, T r) {
  return l + r;
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_sub(T l, T r) {
  return l - r;
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_mul(T l, T r) {
  return l * r;
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_div(T l, T r) {
  return l / r;
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_floor_div(T l, T r) {
  return static_cast<T>(std::trunc(l / r));
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_max(T l, T r) {
  return l > r ? l : r;
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_mod(T l, T r) {
  return l % r;
}

template <typename T>
static inline T __attribute__((__always_inline__)) naive_pow(T l, T r) {
  return std::pow(l, r);
}

template <typename T, T naive_op(T, T)>
static void naive_elementwise_op(const T* dinx,
                                 const T* diny,
                                 T* dout,
                                 int num) {
  int cnt = num >> 2;
  int remain = num % 4;
  for (int i = 0; i < cnt; i++) {
    const T* dinx_ptr = dinx + (i << 2);
    const T* diny_ptr = diny + (i << 2);
    T* dout_ptr = dout + (i << 2);

    T dinx0 = dinx_ptr[0];
    T dinx1 = dinx_ptr[1];
    T dinx2 = dinx_ptr[2];
    T dinx3 = dinx_ptr[3];

    T diny0 = diny_ptr[0];
    T diny1 = diny_ptr[1];
    T diny2 = diny_ptr[2];
    T diny3 = diny_ptr[3];

    dinx0 = naive_op(dinx0, diny0);
    dinx1 = naive_op(dinx1, diny1);
    dinx2 = naive_op(dinx2, diny2);
    dinx3 = naive_op(dinx3, diny3);

    dout_ptr[0] = dinx0;
    dout_ptr[1] = dinx1;
    dout_ptr[2] = dinx2;
    dout_ptr[3] = dinx3;
  }
  if (remain > 0) {
    const T* dinx_ptr = dinx + (cnt << 2);
    const T* diny_ptr = diny + (cnt << 2);
    T* dout_ptr = dout + (cnt << 2);
    T tmp = 0;
    for (int i = 0; i < remain; i++) {
      tmp = naive_op(*dinx_ptr++, *diny_ptr++);
      *dout_ptr++ = tmp;
    }
  }
}

template <typename T, T naive_op(T, T)>
static void naive_elementwise_op_broadcast(const T* x_data,
                                           const T* y_data,
                                           T* out_data,
                                           int batch,
                                           int channels,
                                           int num) {
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const T* din_ptr = x_data + offset;
      const T diny_data = y_data[j];
      T* dout_ptr = out_data + offset;

      int cnt = num >> 2;
      int remain = num % 4;
      for (int k = 0; k < cnt; ++k) {
        T dinx0 = din_ptr[0];
        T dinx1 = din_ptr[1];
        T dinx2 = din_ptr[2];
        T dinx3 = din_ptr[3];
        dinx0 = naive_op(dinx0, diny_data);
        dinx1 = naive_op(dinx1, diny_data);
        dinx2 = naive_op(dinx2, diny_data);
        dinx3 = naive_op(dinx3, diny_data);

        dout_ptr[0] = dinx0;
        dout_ptr[1] = dinx1;
        dout_ptr[2] = dinx2;
        dout_ptr[3] = dinx3;
        din_ptr += 4;
        dout_ptr += 4;
      }
      if (remain > 0) {
        T tmp = 0;
        for (int p = 0; p < remain; p++) {
          tmp = naive_op(*din_ptr++, diny_data);
          *dout_ptr++ = tmp;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
