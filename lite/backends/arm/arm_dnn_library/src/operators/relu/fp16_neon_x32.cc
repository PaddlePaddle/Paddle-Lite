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

#include <arm_neon.h>
#include <assert.h>
#include "arm_dnn_library/core/types.h"
#include "runtime/context.h"
#include "utilities/thread_pool.h"

namespace armdnnlibrary {
namespace kernels {

Status relu_fp16_neon_x32(Context* context,
                          const float16* input_data,
                          float16* output_data,
                          size_t size) {
  assert(input_data != NULL);
  assert(output_data != NULL);
  assert(size != 0);
  int thread_num = context->work_thread_num();
  int size_per_thread = size / thread_num;
  int remain = size - thread_num * size_per_thread;
  int loop_32_per_thread = size_per_thread >> 5;
  int remain_32_per_thread = size_per_thread & 31;
  int loop_8_per_thread = remain_32_per_thread >> 3;
  int remain_8_per_thread = remain_32_per_thread & 7;

  const float16x8_t vzero = vdupq_n_f16(0.0f);
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(i, tid, thread_num) {
    const float16_t* input_ptr_in_thread =
        reinterpret_cast<const float16_t*>(input_data) + i * size_per_thread;
    float16_t* output_ptr_in_thread =
        reinterpret_cast<float16_t*>(output_data) + i * size_per_thread;
    for (int j = 0; j < loop_32_per_thread; j++) {
      float16x8_t vacc0 = vld1q_f16(input_ptr_in_thread);
      input_ptr_in_thread += 8;
      float16x8_t vacc1 = vld1q_f16(input_ptr_in_thread);
      input_ptr_in_thread += 8;
      float16x8_t vacc2 = vld1q_f16(input_ptr_in_thread);
      input_ptr_in_thread += 8;
      float16x8_t vacc3 = vld1q_f16(input_ptr_in_thread);
      input_ptr_in_thread += 8;
      vst1q_f16(output_ptr_in_thread, vmaxq_f16(vacc0, vzero));
      output_ptr_in_thread += 8;
      vst1q_f16(output_ptr_in_thread, vmaxq_f16(vacc1, vzero));
      output_ptr_in_thread += 8;
      vst1q_f16(output_ptr_in_thread, vmaxq_f16(vacc2, vzero));
      output_ptr_in_thread += 8;
      vst1q_f16(output_ptr_in_thread, vmaxq_f16(vacc3, vzero));
      output_ptr_in_thread += 8;
    }
    for (int j = 0; j < loop_8_per_thread; j++) {
      float16x8_t vacc0 = vld1q_f16(input_ptr_in_thread);
      input_ptr_in_thread += 8;
      vst1q_f16(output_ptr_in_thread, vmaxq_f16(vacc0, vzero));
      output_ptr_in_thread += 8;
    }
    for (int j = 0; j < remain_8_per_thread; j++) {
      *output_ptr_in_thread =
          *input_ptr_in_thread > 0.f ? *input_ptr_in_thread : 0.f;
      input_ptr_in_thread++;
      output_ptr_in_thread++;
    }
  }
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_END();
  float16_t* output_ptr =
      reinterpret_cast<float16_t*>(output_data) + thread_num * size_per_thread;
  const float16_t* input_ptr = reinterpret_cast<const float16_t*>(input_data) +
                               thread_num * size_per_thread;
  for (int j = 0; j < remain; j++) {
    *output_ptr = *input_ptr > 0.f ? *input_ptr : 0.f;
    input_ptr++;
    output_ptr++;
  }
  return SUCCESS;
}

}  // namespace kernels
}  // namespace armdnnlibrary
