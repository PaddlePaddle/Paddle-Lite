/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PRELU_OP

#include "operators/kernel/prelu_kernel.h"
#include <operators/math/transform.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename T>
struct PReluFunctor {
  explicit PReluFunctor(float slope) { this->slope_ = slope; }
  inline T operator()(T in) const { return in > 0 ? in : in * slope_; }

  float slope_ = 0.0f;
};

/*
 * @b 特化到具体平台的实现, param 从 op 层传入
 * */
template <>
void PReluKernel<CPU, float>::Compute(const PReluParam<CPU> &param) {
  auto *x = param.InputX();
  auto *alpha = param.InputAlpha();
  auto *out = param.Out();
  std::string mode = param.Mode();
  auto *x_ptr = x->data<float>();
  auto *o_ptr = out->mutable_data<float>();
  auto *alpha_ptr = alpha->data<float>();
  int numel = x->numel();
  auto dim = x->dims();
  int k = dim[0] * dim[1];
  int n = dim[2] * dim[3];
  int index = 0;
  int i = 0;
  int temp = 0;
#if __ARM_NEON
  #pragma omp parallel for
  for (int i = 0; i < k; i++) {
    float32x4_t zero = vdupq_n_f32(0.0);
    float32x4_t cv;
    float32x4_t cv1;
    float32x4_t cv2;
    float32x4_t pv;
    for (int j = 0; (j + 3) < n; j += 4) {
      const float *in = x_ptr + i * n + j;
      float *out = o_ptr + i * n + j;
      cv = vld1q_f32(in);
      cv1 = vmaxq_f32(cv, zero);
      cv2 = vminq_f32(cv, zero);
      if (mode == "channel") {
        cv2 = vmulq_n_f32(cv2, alpha_ptr[i]);
      } else if (mode == "element") {
        pv = vld1q_f32(alpha_ptr + i * n + j);
        cv2 = vmulq_f32(cv2, pv);
      } else {
        cv2 = vmulq_n_f32(cv2, alpha_ptr[0]);
      }
      cv = vaddq_f32(cv1, cv2);
      vst1q_f32(out, cv);
    }
    int j;
    for (j = 0; (j + 3) < n; j += 4) {
    }
    for (int m = j; m < n; m++) {
      if (mode == "channel") {
        o_ptr[i * n + m] = x_ptr[i * n + m] > 0
                               ? x_ptr[i * n + m]
                               : alpha_ptr[i] * x_ptr[i * n + m];
      } else if (mode == "element") {
        o_ptr[i * n + m] = x_ptr[i * n + m] > 0
                               ? x_ptr[i * n + m]
                               : alpha_ptr[i * n + m] * x_ptr[i * n + m];
      } else {
        o_ptr[i * n + m] = x_ptr[i * n + m] > 0
                               ? x_ptr[i * n + m]
                               : alpha_ptr[0] * x_ptr[i * n + m];
      }
    }
  }

#else
  if (mode == "channel") {
    temp = numel / (dim[0] * dim[1]);
#pragma omp parallel for
    for (i = 0; i < numel; i++) {
      index = (i / temp) % dim[1];
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[index] * x_ptr[i];
    }
  } else if (mode == "element") {
#pragma omp parallel for
    for (i = 0; i < numel; i++) {
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[i] * x_ptr[i];
    }
  } else {
#pragma omp parallel for
    for (i = 0; i < numel; i++) {
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[0] * x_ptr[i];
    }
  }
#endif
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
