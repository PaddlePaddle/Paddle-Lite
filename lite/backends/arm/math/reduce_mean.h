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

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void reduce_mean_n(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in);

template <typename T>
void reduce_mean_c(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in);

template <typename T>
void reduce_mean_h(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in);

template <typename T>
void reduce_mean_w(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in);

template <typename T>
void reduce_mean_nc(const T* src,
                    T* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in);

template <typename T>
void reduce_mean_ch(const T* src,
                    T* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in);

template <typename T>
void reduce_mean_hw(const T* src,
                    T* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in);

template <typename T>
void reduce_mean_all(const T* src,
                     T* dst,
                     int num_in,
                     int channel_in,
                     int height_in,
                     int width_in);

template <typename T>
void mean_grad(const T* out_grad, T* in_grad, int size);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
