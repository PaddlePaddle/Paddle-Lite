// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/backends/x86/math/elementwise_common_broadcast_config.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

#define ElementWiseFunc(op)                                                    \
  template <typename T>                                                        \
  void Elementwise_##op(const T* dinx,                                         \
                        const T* diny,                                         \
                        T* dout,                                               \
                        int num,                                               \
                        bool has_active,                                       \
                        std::string act_type) {                                \
    if (act_type == "tanh") {                                                  \
      lite::x86::math::elementwise_range_to_range<                             \
          MergeConfig<op##Config<T>, ActiveConfig<ActiveType::TANH, T>>>(      \
          dinx, diny, dout, num);                                              \
    } else if (act_type == "relu") {                                           \
      lite::x86::math::elementwise_range_to_range<                             \
          MergeConfig<op##Config<T>, ActiveConfig<ActiveType::RELU, T>>>(      \
          dinx, diny, dout, num);                                              \
    } else if (act_type == "sigmoid") {                                        \
      lite::x86::math::elementwise_range_to_range<                             \
          MergeConfig<op##Config<T>, ActiveConfig<ActiveType::SIGMOID, T>>>(   \
          dinx, diny, dout, num);                                              \
    } else {                                                                   \
      lite::x86::math::elementwise_range_to_range<                             \
          MergeConfig<op##Config<T>, ActiveConfig<ActiveType::NO_ACTIVE, T>>>( \
          dinx, diny, dout, num);                                              \
    }                                                                          \
  }

#define ElementWiseFuncBCast(op)                                      \
  template <typename T>                                               \
  void Elementwise_Broadcast_##op(const T* dinx,                      \
                                  const T* diny,                      \
                                  T* dout,                            \
                                  int batch,                          \
                                  int channels,                       \
                                  int num,                            \
                                  bool has_active,                    \
                                  std::string act_type,               \
                                  bool inv) {                         \
    if (act_type == "tanh") {                                         \
      for (int i = 0; i < batch; ++i) {                               \
        for (int j = 0; j < channels; ++j) {                          \
          int offset = (i * channels + j) * num;                      \
          auto* dout_ptr = dout + offset;                             \
          if (inv) {                                                  \
            const auto* dinx_ptr = dinx + j;                          \
            const auto* diny_ptr = diny + offset;                     \
            lite::x86::math::elementwise_one_to_range<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::TANH, T>>>(      \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          } else {                                                    \
            const auto* dinx_ptr = dinx + offset;                     \
            const auto* diny_ptr = diny + j;                          \
            lite::x86::math::elementwise_range_to_one<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::TANH, T>>>(      \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          }                                                           \
        }                                                             \
      }                                                               \
    } else if (act_type == "relu") {                                  \
      for (int i = 0; i < batch; ++i) {                               \
        for (int j = 0; j < channels; ++j) {                          \
          int offset = (i * channels + j) * num;                      \
          auto* dout_ptr = dout + offset;                             \
          if (inv) {                                                  \
            const auto* dinx_ptr = dinx + j;                          \
            const auto* diny_ptr = diny + offset;                     \
            lite::x86::math::elementwise_one_to_range<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::RELU, T>>>(      \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          } else {                                                    \
            const auto* dinx_ptr = dinx + offset;                     \
            const auto* diny_ptr = diny + j;                          \
            lite::x86::math::elementwise_range_to_one<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::RELU, T>>>(      \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          }                                                           \
        }                                                             \
      }                                                               \
    } else if (act_type == "sigmoid") {                               \
      for (int i = 0; i < batch; ++i) {                               \
        for (int j = 0; j < channels; ++j) {                          \
          int offset = (i * channels + j) * num;                      \
          auto* dout_ptr = dout + offset;                             \
          if (inv) {                                                  \
            const auto* dinx_ptr = dinx + j;                          \
            const auto* diny_ptr = diny + offset;                     \
            lite::x86::math::elementwise_one_to_range<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::SIGMOID, T>>>(   \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          } else {                                                    \
            const auto* dinx_ptr = dinx + offset;                     \
            const auto* diny_ptr = diny + j;                          \
            lite::x86::math::elementwise_range_to_one<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::SIGMOID, T>>>(   \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          }                                                           \
        }                                                             \
      }                                                               \
    } else {                                                          \
      for (int i = 0; i < batch; ++i) {                               \
        for (int j = 0; j < channels; ++j) {                          \
          int offset = (i * channels + j) * num;                      \
          auto* dout_ptr = dout + offset;                             \
          if (inv) {                                                  \
            const auto* dinx_ptr = dinx + j;                          \
            const auto* diny_ptr = diny + offset;                     \
            lite::x86::math::elementwise_one_to_range<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::NO_ACTIVE, T>>>( \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          } else {                                                    \
            const auto* dinx_ptr = dinx + offset;                     \
            const auto* diny_ptr = diny + j;                          \
            lite::x86::math::elementwise_range_to_one<                \
                MergeConfig<op##Config<T>,                            \
                            ActiveConfig<ActiveType::NO_ACTIVE, T>>>( \
                dinx_ptr, diny_ptr, dout_ptr, num);                   \
          }                                                           \
        }                                                             \
      }                                                               \
    }                                                                 \
  }

// clang-format off
ElementWiseFunc(Add)
ElementWiseFuncBCast(Add)
ElementWiseFunc(Sub)
ElementWiseFuncBCast(Sub)
ElementWiseFunc(Mul)
ElementWiseFuncBCast(Mul)
ElementWiseFunc(Max)
ElementWiseFuncBCast(Max)
ElementWiseFunc(Min)
ElementWiseFuncBCast(Min)
ElementWiseFunc(Div)
ElementWiseFuncBCast(Div)
ElementWiseFunc(FloorDiv)
ElementWiseFuncBCast(FloorDiv)
ElementWiseFunc(Mod)
ElementWiseFuncBCast(Mod)
ElementWiseFunc(Pow)
ElementWiseFuncBCast(Pow)
// clang-format on

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
