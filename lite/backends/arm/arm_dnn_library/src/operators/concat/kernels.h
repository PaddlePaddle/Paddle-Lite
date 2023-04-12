// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <assert.h>
#include <vector>
#include "arm_dnn_library/core/types.h"
#include "runtime/context.h"
#include "utilities/thread_pool.h"

namespace armdnnlibrary {
namespace kernels {

// Reference implementation
template <typename T>
Status concat(Context* context,
              const std::vector<const T*>& x_datas,
              const std::vector<std::vector<int64_t>>& x_shapes,
              T* y_data,
              int64_t axis) {
  return FEATURE_NOT_SUPPORTED;
}

// Architecture-dependent implementation

}  // namespace kernels
}  // namespace armdnnlibrary
