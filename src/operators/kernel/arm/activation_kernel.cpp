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
#include "operators/kernel/central-arm-func/activation_arm_func.h"
#include "operators/math/activation.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

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

#ifdef LEAKY_RELU_OP
template <>
bool LeakyReluKernel<CPU, float>::Init(LeakyReluParam<CPU> *param) {
  return true;
}

template <>
void LeakyReluKernel<CPU, float>::Compute(const LeakyReluParam<CPU> &param) {
  const LoDTensor *input = param.InputX();
  LoDTensor *output = param.Out();
  ActivationCompute<float, LEAKY_RELU>()(input, output, param.Alpha());
  output->set_lod(input->lod());
}
#endif

}  // namespace operators
}  // namespace paddle_mobile
