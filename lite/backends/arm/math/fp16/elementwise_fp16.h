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
namespace fp16 {
typedef __fp16 float16_t;

#define elementwise_simple_compute_declare(op)                           \
  template <typename T>                                                  \
  void elementwise_##op(const T* dinx, const T* diny, T* dout, int num); \
                                                                         \
  template <typename T>                                                  \
  void elementwise_##op##_relu(                                          \
      const T* dinx, const T* diny, T* dout, int num);                   \
                                                                         \
  template <typename T>                                                  \
  void elementwise_##op##_broadcast(const T* dinx,                       \
                                    const T* diny,                       \
                                    T* dout,                             \
                                    int batch,                           \
                                    int channels,                        \
                                    int num);                            \
                                                                         \
  template <typename T>                                                  \
  void elementwise_##op##_relu_broadcast(const T* dinx,                  \
                                         const T* diny,                  \
                                         T* dout,                        \
                                         int batch,                      \
                                         int channels,                   \
                                         int num);

elementwise_simple_compute_declare(add);
elementwise_simple_compute_declare(mul);
elementwise_simple_compute_declare(sub);
#ifdef __aarch64__
elementwise_simple_compute_declare(div);
#else
void elementwise_div(const float16_t* dinx,
                     const float16_t* diny,
                     float16_t* dout,
                     int num);

void elementwise_div_broadcast(const float16_t* dinx,
                               const float16_t* diny,
                               float16_t* dout,
                               int batch,
                               int channels,
                               int num);
#endif

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
