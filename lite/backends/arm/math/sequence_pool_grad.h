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
void seq_pool_sum_grad(const T* din,
                       const T* dout_grad,
                       T* din_grad,
                       const std::vector<uint64_t> lod,
                       int64_t width);

template <typename T>
void seq_pool_average_grad(const T* din,
                           const T* dout_grad,
                           T* din_grad,
                           const std::vector<uint64_t> lod,
                           int64_t width);

template <typename T>
void seq_pool_sqrt_grad(const T* din,
                        const T* dout_grad,
                        T* din_grad,
                        const std::vector<uint64_t> lod,
                        int64_t width);

template <typename T>
void seq_pool_max_grad(const T* din,
                       const T* dout_grad,
                       const int64_t* index_grad,
                       T* din_grad,
                       const std::vector<uint64_t> lod,
                       int64_t width);

template <typename T>
void seq_pool_first_grad(const T* din,
                         const T* dout_grad,
                         T* din_grad,
                         const std::vector<uint64_t> lod,
                         int64_t width);

template <typename T>
void seq_pool_last_grad(const T* din,
                        const T* dout_grad,
                        T* din_grad,
                        const std::vector<uint64_t> lod,
                        int64_t width);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
