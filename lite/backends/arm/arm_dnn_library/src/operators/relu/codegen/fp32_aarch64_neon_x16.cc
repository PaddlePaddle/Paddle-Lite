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
// Output file: src/operators/relu/codegen/fp32_aarch64_neon_x16.cc
// Command args:  -D BATCH_TILE=16 -D ARCH=aarch64
#include <arm_neon.h>
#include <assert.h>
#include "arm_dnn_library/core/types.h"
#include "runtime/context.h"
#include "utilities/thread_pool.h"

namespace armdnnlibrary {
namespace kernels {

Status relu_fp32_aarch64_neon_x16(Context* context,
                                  const float* input_data,
                                  float* output_data,
                                  size_t size) {
  assert(input_data != NULL);
  assert(output_data != NULL);
  assert(size != 0);
  int thread_num = context->work_thread_num();
  int size_per_thread = size / thread_num;
  int remain = size - thread_num * size_per_thread;
  int loop_per_thread = size_per_thread / 16;
  int remain_per_thread = size_per_thread - (loop_per_thread * 16);

  const float32x4_t vzero = vmovq_n_f32(0.0f);
  ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(i, tid, thread_num) {
    const float* input_ptr_in_thread = input_data + i * size_per_thread;
    float* output_ptr_in_thread = output_data + i * size_per_thread;
    for (int j = 0; j < loop_per_thread; j++) {
      float32x4_t vacc0 = vld1q_f32(input_ptr_in_thread);
      input_ptr_in_thread += 4;
      float32x4_t vacc1 = vld1q_f32(input_ptr_in_thread);
      input_ptr_in_thread += 4;
      float32x4_t vacc2 = vld1q_f32(input_ptr_in_thread);
      input_ptr_in_thread += 4;
      float32x4_t vacc3 = vld1q_f32(input_ptr_in_thread);
      input_ptr_in_thread += 4;

      vacc0 = vmaxq_f32(vacc0, vzero);
      vacc1 = vmaxq_f32(vacc1, vzero);
      vacc2 = vmaxq_f32(vacc2, vzero);
      vacc3 = vmaxq_f32(vacc3, vzero);

      vst1q_f32(output_ptr_in_thread, vacc0);
      output_ptr_in_thread += 4;
      vst1q_f32(output_ptr_in_thread, vacc1);
      output_ptr_in_thread += 4;
      vst1q_f32(output_ptr_in_thread, vacc2);
      output_ptr_in_thread += 4;
      vst1q_f32(output_ptr_in_thread, vacc3);
      output_ptr_in_thread += 4;
    }
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
