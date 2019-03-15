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

#include "operators/kernel/activation_kernel.h"
#include "common/types.h"
#include "framework/context.h"
#include "operators/math/activation.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename Dtype, ActivationType Act>
struct ActivationCompute {
  void operator()(const Tensor *input, Tensor *output) {}
};

template <ActivationType Act>
struct ActivationCompute<float, Act> {
  void operator()(const Tensor *input, Tensor *output) {
    const float *x = input->data<float>();
    float *y = output->mutable_data<float>();
    size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    size_t loop = remain >> 4;
    remain = remain & 0xF;

#pragma omp parallel for num_threads(framework::threads())
    for (size_t i = 0; i < loop; ++i) {
      const float *local_x = x + (i << 4);
      float *local_y = y + (i << 4);
      float32x4_t r0 = vld1q_f32(local_x);
      float32x4_t r1 = vld1q_f32(local_x + 4);
      float32x4_t r2 = vld1q_f32(local_x + 8);
      float32x4_t r3 = vld1q_f32(local_x + 12);
      r0 = math::vActiveq_f32<Act>(r0);
      r1 = math::vActiveq_f32<Act>(r1);
      r2 = math::vActiveq_f32<Act>(r2);
      r3 = math::vActiveq_f32<Act>(r3);
      vst1q_f32(local_y, r0);
      vst1q_f32(local_y + 4, r1);
      vst1q_f32(local_y + 8, r2);
      vst1q_f32(local_y + 12, r3);
    }
    x += (loop << 4);
    y += (loop << 4);
#endif
    for (size_t i = 0; i < remain; ++i) {
      y[i] = math::Active<Act>(x[i]);
    }
  }
};

#ifdef RELU_OP
template <>
bool ReluKernel<CPU, float>::Init(ReluParam<CPU> *param) {
  return true;
}

template <>
void ReluKernel<CPU, float>::Compute(const ReluParam<CPU> &param) {
  const LoDTensor *input = param.InputX();
  LoDTensor *output = param.Out();
  ActivationCompute<float, RELU>()(input, output);
  output->set_lod(input->lod());
}

template <>
bool Relu6Kernel<CPU, float>::Init(ReluParam<CPU> *param) {
  return true;
}

template <>
void Relu6Kernel<CPU, float>::Compute(const ReluParam<CPU> &param) {
  const LoDTensor *input = param.InputX();
  LoDTensor *output = param.Out();
  ActivationCompute<float, RELU6>()(input, output);
  output->set_lod(input->lod());
}
#endif

#ifdef SIGMOID_OP
template <>
bool SigmoidKernel<CPU, float>::Init(SigmoidParam<CPU> *param) {
  return true;
}

template <>
void SigmoidKernel<CPU, float>::Compute(const SigmoidParam<CPU> &param) {
  const LoDTensor *input = param.InputX();
  LoDTensor *output = param.Out();
  ActivationCompute<float, SIGMOID>()(input, output);
  output->set_lod(input->lod());
}
#endif

#ifdef TANH_OP
template <>
bool TanhKernel<CPU, float>::Init(TanhParam<CPU> *param) {
  return true;
}

template <>
void TanhKernel<CPU, float>::Compute(const TanhParam<CPU> &param) {
  const LoDTensor *input = param.InputX();
  LoDTensor *output = param.Out();
  ActivationCompute<float, TANH>()(input, output);
  output->set_lod(input->lod());
}
#endif

#ifdef LOG_OP
template <>
bool LogKernel<CPU, float>::Init(ReluParam<CPU> *param) {
  return true;
}

template <>
void LogKernel<CPU, float>::Compute(const ReluParam<CPU> &param) {
  const LoDTensor *input = param.InputX();
  LoDTensor *output = param.Out();
  ActivationCompute<float, LOG>()(input, output);
  output->set_lod(input->lod());
}
#endif

}  // namespace operators
}  // namespace paddle_mobile
