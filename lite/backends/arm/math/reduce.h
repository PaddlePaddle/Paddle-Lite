/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

enum class ReduceProcessType { mean, sum, max, min, prod };

template <typename T>
void ReduceImpl(const T* X,
                const std::vector<int64_t>& x_dims,
                T* Out,
                const std::vector<int64_t>& out_dims,
                const std::vector<int>& dim,
                bool reduce_all,
                ReduceProcessType op_name);

template <typename T>
void mean_grad(const T* out_grad, T* in_grad, int size);
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
