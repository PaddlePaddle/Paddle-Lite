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

#include <algorithm>
#include <string>
#include <vector>
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
template <typename T>
void permute_basic(const int count,
                   const T* din,
                   const int* permute_order,
                   const int* old_steps,
                   const int* new_steps,
                   const int num_axes,
                   T* dout);
template <typename T>
void transpose_mat(
    const T* din, T* dout, const int num, const int width, const int height);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
