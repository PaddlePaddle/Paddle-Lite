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

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void elementwise_add(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_add_relu(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void elementwise_add_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void elementwise_add_relu_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

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

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
