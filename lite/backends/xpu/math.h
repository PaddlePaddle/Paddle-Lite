// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdint.h>
#include <cmath>
#include <cstdlib>
#include <utility>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace xpu {
namespace math {

static inline long round_half_to_even(const float src) {  // NOLINT
  long ret = llround(src);                                // NOLINT
  if (fabs(fabs(round(src) - src) - 0.5) > 0) {
    return ret;
  } else {
    if (abs(ret) % 2 == 0) {
      return ret;
    } else {
      return ret + (ret > 0 ? -1 : 1);
    }
  }
}

static float ieee_compliance_0(float f) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(&f);
  uint32_t sign = (*ptr) & 0x80000000;
  uint32_t uf = 0;
  // nan -> inf
  if (std::isnan(f)) {
    uf = (sign | 0x7F800000);
    float *ptr = reinterpret_cast<float *>(&uf);
    return *ptr;
  } else if (std::isnormal(f) || (std::isinf(f)) || (f == 0)) {
    return f;
  } else {
    // denormal -> +-0
    uf = 0x0;
    float *ptr = reinterpret_cast<float *>(&uf);
    return *ptr;
  }
}

template <typename T, int RMAX>
static inline T fp32_to_intx(const float f, float max) {
  max = ieee_compliance_0(max);
  float input = ieee_compliance_0(f);
  // +0 and -0 -> +0
  if (input == 0) {
    input = 0.0f;
  }

  float tmp = RMAX / max;
  if (std::isinf(tmp)) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(&input);
    if ((*ptr) >> 31 & 1) {
      return T(-RMAX);
    } else {
      return T(RMAX);
    }
  }

  tmp = input * tmp;
  if (std::isnan(tmp)) {
    return T(RMAX);
  }

  tmp = ieee_compliance_0(tmp);
  // early check to avoid INF or big value get into convertor func.
  if (tmp > RMAX) {
    return T(RMAX);
  }
  if (tmp < -RMAX) {
    return T(-RMAX);
  }
  T ret = (T)round_half_to_even(tmp);
  if (ret > RMAX) {
    ret = T(RMAX);
  }
  if (ret < -RMAX) {
    ret = T(-RMAX);
  }
  return ret;
}

static inline int16_t fp32_to_int16(const float f, float max) {
  int16_t v1 = fp32_to_intx<int16_t, 32767>(f, max);
  return v1;
}

static inline int ConvertFP32ToInt16(const void *input,
                                     void *output,
                                     float max_val,
                                     int len) {
  for (int i = 0; i < len; i++) {
    static_cast<int16_t *>(output)[i] =
        fp32_to_int16(static_cast<const float *>(input)[i], max_val);
  }
  return 0;
}

static inline int8_t fp32_to_int8(const float f, float max) {
  int8_t v1 = fp32_to_intx<int8_t, 127>(f, max);
  return v1;
}

static inline int ConvertFP32ToInt8(const void *input,
                                    void *output,
                                    float max_val,
                                    int len) {
  for (int i = 0; i < len; i++) {
    static_cast<int8_t *>(output)[i] =
        fp32_to_int8(static_cast<const float *>(input)[i], max_val);
  }
  return 0;
}

#ifdef LITE_WITH_XPU
static inline int ConvertFP32ToFP16(const void *input, void *output, int len) {
  for (int i = 0; i < len; i++) {
    static_cast<float16 *>(output)[i] =
        float16(static_cast<const float *>(input)[i]);
  }
  return 0;
}
#endif

static inline float FindMaxAbs(const float *data, int len) {
  float max_f = 0.0f;
  for (int i = 0; i < len; ++i) {
    float max = std::abs(data[i]);
    if (max > max_f) {
      max_f = max;
    }
  }
  return max_f;
}

template <typename T>
static inline void Transpose(const T *in, T *out, int h, int w) {
  for (int h1 = 0; h1 < w; ++h1) {
    for (int w1 = 0; w1 < h; ++w1) {
      out[h1 * h + w1] = in[w1 * w + h1];
    }
  }
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static lite::DDim RowMatrixFromVector(const lite::DDim &x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return lite::DDim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the rank of y_dim > 1, the
 * original y_dim is returned.
 */
static lite::DDim ColumnMatrixFromVector(const lite::DDim &y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return lite::DDim({y_dim[0], 1});
}

/**
 * Matrix Descriptor of a memory buffer.
 *
 * It is used for Blas::MatMul. MatMul operator can be batched.
 * if Mat A is [BatchSize, H, W], Mat B is [BatchSize, H, W]. It will be a
 * `batch_size` times of GEMM. The batched GEMM could be faster base on the
 * implementation of the blas library. The batch size could be zero. If any
 * matrix of `matmul` has a batch size, the will be a batched GEMM, too. e.g.,
 * Mat A is [BatchSize, H1, W2], and Mat B [H2, W2], The result matrix wil be
 * [BatchSize, H1, W2]
 *
 * The boolean flag, `trans`, describe the memory is the transpose of matrix or
 * not. If the trans is true, the last two dims of matrix are transposed. The
 * memory layout of the matrix is [Width, Height] or [BatchSize, Width, Height].
 *
 * The MatDescriptor is not only the dimension or shape of a matrix, it also
 * contains the layout, stride of matrix. It is clearer to have a structure than
 * reuse `DDim`.
 */
struct MatDescriptor {
  int64_t height_;
  int64_t width_;
  int64_t stride_{0};
  int64_t batch_size_{0};
  bool trans_;
};

static MatDescriptor CreateMatrixDescriptor(const lite::DDimLite &tensor_dim,
                                            int num_flatten_cols,
                                            bool trans) {
  CHECK_GT(tensor_dim.size(), 1u);
  MatDescriptor retv;
  if (num_flatten_cols > 1) {
    auto flatten_dim = tensor_dim.Flatten2D(num_flatten_cols);
    retv.height_ = flatten_dim[0];
    retv.width_ = flatten_dim[1];
  } else {
    if (tensor_dim.size() == 2) {
      retv.height_ = tensor_dim[0];
      retv.width_ = tensor_dim[1];
    } else {
      auto dim_vec = tensor_dim.Vectorize();
      retv.batch_size_ = 1;
      for (size_t i = 0; i < dim_vec.size() - 2; ++i) {
        retv.batch_size_ *= dim_vec[i];
      }
      retv.height_ = dim_vec[dim_vec.size() - 2];
      retv.width_ = dim_vec[dim_vec.size() - 1];
      retv.stride_ = retv.height_ * retv.width_;
    }
  }
  if (trans) {
    std::swap(retv.width_, retv.height_);
  }
  retv.trans_ = trans;
  return retv;
}

}  // namespace math
}  // namespace xpu
}  // namespace lite
}  // namespace paddle
