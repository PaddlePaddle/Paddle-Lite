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

#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace xpu {
namespace math {

static inline float FindMaxAbs(const float* data, int len) {
  float max_f = 0.0f;
  for (int i = 0; i < len; ++i) {
    float max = std::abs(data[i]);
    if (max > max_f) {
      max_f = max;
    }
  }
  return max_f;
}

template<typename T>
static inline void Transpose(const T* in, T* out, int h, int w) {
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
