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
namespace fp16 {

template <typename T>
void act_relu(const T* din, T* dout, int size, int threads);

template <typename T>
void act_hard_sigmoid(const T* din,
                      T* dout,
                      const int size,
                      const float slope,
                      const float offset,
                      int threads);

template <typename T>
void act_hard_swish(const T* din,
                    T* dout,
                    const int size,
                    const float threshold,
                    const float scale,
                    const float offset,
                    int threads);
template <typename T>
void act_prelu(const T* din,
               T* dout,
               int outer_size,
               int channel_size,
               int inner_size,
               std::string mode,
               const T* alpha_data,
               int threads);

template <typename T>
void act_tanh(const T* din, T* dout, int size, int threads);

template <typename T>
void act_sigmoid(const T* din, T* dout, int size, int threads);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
