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

#ifdef RELU_OP

#include "operators/kernel/relu_kernel.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

enum ReluMode {
  Relu = 0,
  Relu6 = 1,
  PRelu = 2,
  LeakyRelu = 3,
};

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <ReluMode R = Relu>
inline float32x4_t vRelu_f32(const float32x4_t &x) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  return vmaxq_f32(__zero, x);
}

template <>
inline float32x4_t vRelu_f32<Relu6>(const float32x4_t &x) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  float32x4_t __six = vdupq_n_f32(6.f);
  return vminq_f32(__six, vmaxq_f32(__zero, x));
}
#endif

template <ReluMode R = Relu>
inline float ReluFunc(const float &x) {
  return std::max(x, 0.f);
}

template <>
inline float ReluFunc<Relu6>(const float &x) {
  return std::min(std::max(x, 0.f), 6.f);
}

template <typename Dtype, ReluMode R>
struct ReluCompute {
  void operator()(const Tensor *input, Tensor *output) {}
};

template <ReluMode R>
struct ReluCompute<float, R> {
  void operator()(const Tensor *input, Tensor *output) {
    const float *x = input->data<float>();
    float *y = output->mutable_data<float>();
    size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    size_t loop = remain >> 4;
    remain = remain & 0xF;

    #pragma omp parallel for
    for (size_t i = 0; i < loop; ++i) {
      const float *local_x = x + (i << 4);
      float *local_y = y + (i << 4);
      float32x4_t r0 = vld1q_f32(local_x);
      float32x4_t r1 = vld1q_f32(local_x + 4);
      float32x4_t r2 = vld1q_f32(local_x + 8);
      float32x4_t r3 = vld1q_f32(local_x + 12);
      r0 = vRelu_f32<R>(r0);
      r1 = vRelu_f32<R>(r1);
      r2 = vRelu_f32<R>(r2);
      r3 = vRelu_f32<R>(r3);
      vst1q_f32(local_y, r0);
      vst1q_f32(local_y + 4, r1);
      vst1q_f32(local_y + 8, r2);
      vst1q_f32(local_y + 12, r3);
    }
    x += (loop << 4);
    y += (loop << 4);
#endif
    for (size_t i = 0; i < remain; ++i) {
      y[i] = ReluFunc<R>(x[i]);
    }
  }
};

template <>
bool ReluKernel<CPU, float>::Init(ReluParam<CPU> *param) {
  return true;
}

template <>
void ReluKernel<CPU, float>::Compute(const ReluParam<CPU> &param) {
  const Tensor *input = param.InputX();
  Tensor *output = param.Out();
  ReluCompute<float, Relu>()(input, output);
}

template <>
bool Relu6Kernel<CPU, float>::Init(ReluParam<CPU> *param) {
  return true;
}

template <>
void Relu6Kernel<CPU, float>::Compute(const ReluParam<CPU> &param) {
  const Tensor *input = param.InputX();
  Tensor *output = param.Out();
  ReluCompute<float, Relu6>()(input, output);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
