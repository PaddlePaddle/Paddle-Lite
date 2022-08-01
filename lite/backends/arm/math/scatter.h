/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <stdint.h>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename Dtype>
void scatter(const Dtype* indexs,
             const float* updates,
             float* dst,
             int index_size,
             int num,
             int size,
             bool overwrite) {
  memset(reinterpret_cast<char*>(dst), 0, sizeof(float) * size * num);
  if (overwrite) {
    for (int i = 0; i < index_size; i++) {
      const float* din = updates + i * size;
      memcpy(dst + indexs[i] * size, din, sizeof(float) * size);
    }
  } else {
    int cnt = size >> 3;
    int rem = size & 7;
    for (int i = 0; i < index_size; i++) {
      const float* din = updates + i * size;
      float* dout = dst + indexs[i] * size;
      for (int j = 0; j < cnt; j++) {
        float32x4_t va0 = vld1q_f32(din);
        float32x4_t vb0 = vld1q_f32(dout);
        float32x4_t va1 = vld1q_f32(din + 4);
        float32x4_t vb1 = vld1q_f32(dout + 4);
        vb0 = vaddq_f32(va0, vb0);
        vb1 = vaddq_f32(va1, vb1);
        din += 8;
        vst1q_f32(dout, vb0);
        vst1q_f32(dout + 4, vb1);
        dout += 8;
      }
      for (int j = 0; j < rem; j++) {
        dout[0] += *din++;
        dout++;
      }
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
