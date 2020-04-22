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
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename dtype>
void scale_compute_basic(const operators::ScaleParam& param) {
  const dtype* x_data = param.x->data<dtype>();
  dtype* output_data = param.output->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  DDim output_dims = param.output->dims();
  bool bias_after_scale = param.bias_after_scale;
  float scale = param.scale;
  float bias = param.bias;
  if (!bias_after_scale) {
    bias *= scale;
  }
  for (int i = 0; i < output_dims.production(); i++) {
    output_data[i] = static_cast<dtype>(x_data[i] * scale + bias);
  }
}

template <typename T>
void scale(const T* din, T* dout, int num, T scale, T bias);

template <typename T>
void scale_relu(const T* din, T* dout, int num, T scale, T bias);

template <typename T>
void scale_relu6(const T* din, T* dout, int num, T scale, T bias, T alpha);

template <typename T>
void scale_leaky_relu(const T* din, T* dout, int num, T scale, T bias, T alpha);

template <typename T>
void scale(const T* din,
           T* dout,
           int outer_dim,
           int scale_dim,
           int inner_dim,
           const float* scale_data,
           const float* bias_data);

template <typename T>
void scale(const T* din,
           T* dout,
           int outer_dim,
           int scale_dim,
           const float* scale_data,
           const float* bias_data);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
