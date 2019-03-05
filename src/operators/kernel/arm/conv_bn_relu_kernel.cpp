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

#ifdef FUSION_CONVBNRELU_OP

#include "operators/kernel/conv_bn_relu_kernel.h"
#include <cmath>
#include "operators/kernel/central-arm-func/conv_bn_relu_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNReluKernel<CPU, float>::Init(FusionConvBNReluParam<CPU> *param) {
  const Tensor *mean = param->InputMean();
  const Tensor *variance = param->InputVariance();
  const Tensor *scale = param->InputScale();
  const Tensor *bias = param->InputBias();
  const float epsilon = param->Epsilon();

  //   DLOG << "variance: " << *variance;

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  const int C = mean->numel();
  float inv_std_ptr[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }
  Tensor *new_scale = param->CreateNewScale<Tensor>();
  Tensor *new_bias = param->CreateNewBiase<Tensor>();
  auto new_scale_ptr = new_scale->mutable_data<float>({C});
  auto new_bias_ptr = new_bias->mutable_data<float>({C});
  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
  }

  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);
  return true;
}

template <>
void ConvBNReluKernel<CPU, float>::Compute(
    const FusionConvBNReluParam<CPU> &param) {
  ConvBNReluCompute<float>(param);
}
template class ConvBNReluKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
