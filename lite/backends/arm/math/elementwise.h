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
#include <algorithm>
#include <string>
#include <vector>
#include "lite/operators/op_params.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {
template <typename T>
void elementwise_broadcast_common(T const* x_data,
                                  T const* y_data,
                                  T* out_data,
                                  std::vector<int64_t> x_real_dim,
                                  std::vector<int64_t> y_real_dim,
                                  std::vector<int64_t> out_real_dim,
                                  std::string type,
                                  bool is_xsize_large = false) {
  int out_size = 1;
  int max_dim = out_real_dim.size();
  std::vector<int> index_array(max_dim, 0);
  for (int i = 0; i < max_dim; ++i) {
    out_size *= out_real_dim[i];
  }
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = 0;
    for (int i = 0; i < max_dim; i++) {
      if (x_real_dim[i] > 1) {
        x_index = x_index * x_real_dim[i] + index_array[i];
      }
    }
    y_index = 0;
    for (int i = 0; i < max_dim; i++) {
      if (y_real_dim[i] > 1) {
        y_index = y_index * y_real_dim[i] + index_array[i];
      }
    }

    if (type == "add") {
      out_data[out_index] = x_data[x_index] + y_data[y_index];
    }
    if (type == "mul") {
      out_data[out_index] = x_data[x_index] * y_data[y_index];
    }
  }
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_real_dim[i]) {
      index_array[i] -= out_real_dim[i];
    } else {
      break;
    }
  }
}
template <typename dtype>
void elementwise_compute_basic(const operators::ElementwiseParam& param,
                               const std::string elt_type,
                               const std::string act_type) {
  const dtype* x_data = param.X->data<const dtype>();
  const dtype* y_data = param.Y->data<const dtype>();
  dtype* out_data = param.Out->mutable_data<dtype>();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  // do elementwise add/sub/max...
  if (elt_type == "add") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr + diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "sub") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr - diny_data;
          dout_ptr++;
        }
      }
    }
  } else if (elt_type == "mul") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr * diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "max") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = std::max(*din_ptr, diny_data);
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else {
    LOG(FATAL) << "unsupported Elementwise type: " << elt_type;
  }
  // do activation relu/sigmod...
  if (act_type.size() > 0) {
    if (act_type == "relu") {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          dtype* dout_ptr = out_data + (i * channels + j) * num;
          for (int k = 0; k < num; ++k) {
            *dout_ptr = *dout_ptr > 0.0f ? *dout_ptr : 0.0f;
            dout_ptr++;
          }
        }
      }
    } else {
      LOG(FATAL) << "unsupported Activation type: " << elt_type;
    }
  }
}

template <typename T>
void elementwise_add(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_add_relu(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_add_tanh(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_add_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_add_relu_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_add_grad(const T* dout, T* dinx, int num);

template <typename T>
void elementwise_add_grad_broadcast(
    const T* dout_grad, T* x_grad, T* y_grad, int pre, int n, int post);

template <typename T>
void elementwise_sub(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_sub_relu(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_sub_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_sub_relu_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_sub_grad(const T* dout, T* dinx, T* diny, int num);

template <typename T>
void elementwise_sub_grad_broadcast(
    const T* dout_grad, T* x_grad, T* y_grad, int pre, int n, int post);

template <typename T>
void elementwise_mul(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_mul_relu(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_mul_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_mul_relu_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_max(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_max_relu(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_max_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_max_relu_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_div(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_div_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_div_relu(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_div_relu_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_mod(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_mod_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_pow(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_pow_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
