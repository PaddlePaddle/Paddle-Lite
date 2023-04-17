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

// This file was automatically generated, Do not edit it!
// Template file: src/operators/relu/fp32_neon.cc.in
// Output file: src/operators/relu/codegen/fp32_aarch32_neon_x8.cc
// Command args:  -D BATCH_TILE=8 -D ARCH=aarch32
#include <arm_neon.h>
#include <assert.h>
#include "arm_dnn_library/core/types.h"
#include "runtime/context.h"
#include "utilities/thread_pool.h"

namespace armdnnlibrary {
namespace kernels {

Status relu_fp32_aarch32_neon_x8(Context* context,
                                 const float* input_data,
                                 float* output_data,
                                 size_t size) {
  assert(input_data != NULL);
  assert(output_data != NULL);
  assert(size != 0);
  int thread_num = context->work_thread_num();
  int size_per_thread = size / thread_num;
  int remain = size - thread_num * size_per_thread;
  int loop_per_thread = size_per_thread / 8;
  int remain_per_thread = size_per_thread - (loop_per_thread * 8);

  const float32x4_t vzero = vmovq_n_f32(0.0f);
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(i, tid, thread_num) {
    const float* input_ptr_in_thread = input_data + i * size_per_thread;
    float* output_ptr_in_thread = output_data + i * size_per_thread;
    int loop_in_thread = loop_per_thread;
    asm volatile(
        "1: \n"
        "vld1.32  {d0-d3}, [%[din]]! \n"
        "vld1.32  {d4-d7}, [%[din]]! \n"

        "vmax.f32 q8, q0, %q[vzero] \n"
        "vmax.f32 q9, q1, %q[vzero] \n"
        "vmax.f32 q10, q2, %q[vzero] \n"
        "vmax.f32 q11, q3, %q[vzero] \n"

        "vst1.32  {d16-d19}, [%[dout]]! \n"
        "vst1.32  {d20-d23}, [%[dout]]! \n"

        "subs %[cnt], #1 \n"
        "bne    1b \n"
        : [dout] "+r"(output_ptr_in_thread),
          [din] "+r"(input_ptr_in_thread),
          [cnt] "+r"(loop_per_thread)
        : [vzero] "w"(vzero)
        : "cc", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "memory");
    for (int j = 0; j < remain_per_thread; j++) {
      *output_ptr_in_thread =
          *input_ptr_in_thread > 0.f ? *input_ptr_in_thread : 0.f;
      input_ptr_in_thread++;
      output_ptr_in_thread++;
    }
  }
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_END();
  float* output_ptr = output_data + thread_num * size_per_thread;
  const float* input_ptr = input_data + thread_num * size_per_thread;
  for (int j = 0; j < remain; j++) {
    *output_ptr = *input_ptr > 0.f ? *input_ptr : 0.f;
    input_ptr++;
    output_ptr++;
  }
  return SUCCESS;
}

}  // namespace kernels
}  // namespace armdnnlibrary
