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
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void seq_pool_sum(const T* din,
                  T* dout,
                  const std::vector<uint64_t> lod,
                  int64_t width,
                  T pad_value);

template <typename T>
void seq_pool_average(const T* din,
                      T* dout,
                      const std::vector<uint64_t> lod,
                      int64_t width,
                      T pad_value);

template <typename T>
void seq_pool_sqrt(const T* din,
                   T* dout,
                   const std::vector<uint64_t> lod,
                   int64_t width,
                   T pad_value);

template <typename T>
void seq_pool_max(const T* din,
                  T* dout,
                  int64_t* index,
                  const std::vector<uint64_t> lod,
                  int64_t width,
                  T pad_value);

template <typename T>
void seq_pool_min(const T* din,
                  T* dout,
                  int64_t* index,
                  const std::vector<uint64_t> lod,
                  int64_t width,
                  T pad_value);

template <typename T>
void seq_pool_first(const T* din,
                    T* dout,
                    const std::vector<uint64_t> lod,
                    int64_t width,
                    T pad_value);

template <typename T>
void seq_pool_last(const T* din,
                   T* dout,
                   const std::vector<uint64_t> lod,
                   int64_t width,
                   T pad_value);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
