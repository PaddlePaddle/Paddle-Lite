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
#include <string>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void act_relu(const T* din, T* dout, int size, int threads);

template <typename T>
void act_relu_neg(
    const T* din, T* dout, int size, float negative_slope, int threads);

template <typename T>
void act_clipped_relu(const T* din, T* dout, int size, float coef, int threads);

template <typename T>
void act_prelu(const T* din,
               T* dout,
               int outer_size,
               int channel_size,
               int inner_size,
               std::string mode,
               const float* alpha_data,
               int threads);

template <typename T>
void act_sigmoid(const T* din, T* dout, int size, int threads);

template <typename T>
void act_tanh(const T* din, T* dout, int size, int threads);

template <typename T>
void act_swish(const T* din, T* dout, int size, float coef, int threads);

template <typename T>
void act_log(const T* din, T* dout, int size, int threads);

template <typename T>
void act_exp(const T* din, T* dout, int size, int threads);

template <typename T>
void act_floor(const T* din, T* dout, int size, int threads);

template <typename T>
void act_hard_sigmoid(const T* din,
                      T* dout,
                      const int64_t size,
                      const float slope,
                      const float offset,
                      int threads);

template <typename T>
void act_rsqrt(const T* din, T* dout, int size, int threads);

template <typename T>
void act_square(const T* din, T* dout, int size, int threads);

template <typename T>
void act_hard_swish(const T* din,
                    T* dout,
                    int size,
                    float threshold,
                    float scale,
                    float offset,
                    int threads);
template <typename T>
void act_reciprocal(const T* din, T* dout, int size, int threads);

template <typename T>
void act_abs(const T* din, T* dout, int size, int threads);

template <typename T>
void act_thresholded_relu(
    const T* din, T* dout, int size, float threshold, int threads);

template <typename T>
void act_elu(const T* din, T* dout, int size, float alpha, int threads);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
