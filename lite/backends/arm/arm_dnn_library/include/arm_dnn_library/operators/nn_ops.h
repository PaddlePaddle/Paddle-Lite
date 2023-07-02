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

#include <vector>
#include "arm_dnn_library/core/types.h"

namespace armdnnlibrary {

template <typename T>
Status relu(void* context, const T* x_data, T* y_data, size_t size);
// 8-bit symmetric quantization
Status relu_qs8(void* context,
                const int8_t* x_data,
                float x_scale,
                int8_t* y_data,
                float y_scale,
                size_t size);

}  // namespace armdnnlibrary
