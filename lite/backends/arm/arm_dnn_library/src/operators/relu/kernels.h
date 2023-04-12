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
Status relu(Context* context,
            const T* input_data,
            T* output_data,
            size_t size) {
  assert(input_data != NULL);
  assert(output_data != NULL);
  assert(size != 0);
  int thread_num = context->work_thread_num();
  int size_per_thread = size / thread_num;
  int remain = size - thread_num * size_per_thread;
  T zero = static_cast<T>(0.0);
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(i, tid, thread_num) {
    const T* input_ptr_in_thread = input_data + i * size_per_thread;
    T* output_ptr_in_thread = output_data + i * size_per_thread;
    for (int j = 0; j < size_per_thread; j++) {
      *output_ptr_in_thread =
          *input_ptr_in_thread > zero ? *input_ptr_in_thread : zero;
      input_ptr_in_thread++;
      output_ptr_in_thread++;
    }
  }
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_END();
  T* output_ptr = output_data + thread_num * size_per_thread;
  const T* input_ptr = input_data + thread_num * size_per_thread;
  for (int j = 0; j < remain; j++) {
    *output_ptr = *input_ptr > zero ? *input_ptr : zero;
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
Status relu_fp16_neon_x32(Context* context,
                          const float16* input_data,
                          float16* output_data,
                          size_t size);

}  // namespace kernels
}  // namespace armdnnlibrary
