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
namespace host {
namespace math {

struct LogicalAnd {
  inline bool operator()(const bool a, const bool b) { return a && b; }
};

struct LogicalOr {
  inline bool operator()(const bool a, const bool b) { return a || b; }
};

template <typename T, typename Functor>
void reduce_n(const T* src,
              T* dst,
              int num_in,
              int channel_in,
              int height_in,
              int width_in);

template <typename T, typename Functor>
void reduce_c(const T* src,
              T* dst,
              int num_in,
              int channel_in,
              int height_in,
              int width_in);

template <typename T, typename Functor>
void reduce_h(const T* src,
              T* dst,
              int num_in,
              int channel_in,
              int height_in,
              int width_in);

template <typename T, typename Functor>
void reduce_w(const T* src,
              T* dst,
              int num_in,
              int channel_in,
              int height_in,
              int width_in);

template <typename T, typename Functor>
void reduce_nc(const T* src,
               T* dst,
               int num_in,
               int channel_in,
               int height_in,
               int width_in);

template <typename T, typename Functor>
void reduce_ch(const T* src,
               T* dst,
               int num_in,
               int channel_in,
               int height_in,
               int width_in);

template <typename T, typename Functor>
void reduce_hw(const T* src,
               T* dst,
               int num_in,
               int channel_in,
               int height_in,
               int width_in);

template <typename T, typename Functor>
void reduce_all(const T* src, T* dst, int num_all);

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
