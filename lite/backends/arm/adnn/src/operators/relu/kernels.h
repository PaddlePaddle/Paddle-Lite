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

// Auto-generated file from src/relu/fp32_neon.cc.in, Don't edit it!
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
#include "adnn/core/types.h"  // NOLINT

namespace adnn {
namespace kernels {

// Reference implementation
template <typename T>
ADNN_DLL_EXPORT Status
relu(Context* context, const T* input_data, T* output_data, size_t size) {
  assert(input_data != NULL);
  assert(output_data != NULL);
  assert(size != 0);
  int thread_num = context->thread_num;
  int size_per_thread = size / thread_num;
  int remain = size - thread_num * size_per_thread;

  ADNN_THREAD_POOL_COMMON_TASK_BEGIN(i, tid, thread_num) {
    const float* input_ptr_in_thread = input_data + i * size_per_thread;
    float* output_ptr_in_thread = output_data + i * size_per_thread;
    for (int j = 0; j < size_per_thread; j++) {
      *output_ptr_in_thread =
          *input_ptr_in_thread > 0.f ? *input_ptr_in_thread : 0.f;
      input_ptr_in_thread++;
      output_ptr_in_thread++;
    }
  }
  ADNN_THREAD_POOL_COMMON_TASK_END();
  float* output_ptr = output_data + thread_num * size_per_thread;
  const float* input_ptr = input_data + thread_num * size_per_thread;
  for (int j = 0; j < remain; j++) {
    *output_ptr = *input_ptr > 0.f ? *input_ptr : 0.f;
    input_ptr++;
    output_ptr++;
  }
  return SUCCESS;
}

// Architecture-dependent implementation
Status relu_fp32_aarch32_neon_x8(Context* context,
                                 const float* input_data,
                                 float* output_data,
                                 size_t size);
Status relu_fp32_aarch64_neon_x16(Context* context,
                                  const float* input_data,
                                  float* output_data,
                                  size_t size);

}  // namespace kernels
}  // namespace adnn
