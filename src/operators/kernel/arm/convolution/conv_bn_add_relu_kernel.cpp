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

#ifdef FUSION_CONVBNADDRELU_OP

#include "operators/kernel/conv_bn_add_relu_kernel.h"
#include <cmath>
#include "operators/kernel/arm/convolution/conv_common.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"
#include "operators/math/element_wise.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNAddReluKernel<CPU, float>::Init(
    FusionConvBNAddReluParam<CPU> *param) {
  const Tensor *mean = param->InputMean();
  const Tensor *variance = param->InputVariance();
  const Tensor *scale = param->InputScale();
  const Tensor *bias = param->InputBias();
  const float epsilon = param->Epsilon();

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = const_cast<float *>(scale->data<float>());
  auto bias_ptr = const_cast<float *>(bias->data<float>());

  for (int c = 0; c < scale->numel(); ++c) {
    float inv_scale = 1.f / (pow(variance_ptr[c] + epsilon, 0.5));
    bias_ptr[c] -= inv_scale * scale_ptr[c] * mean_ptr[c];
    scale_ptr[c] *= inv_scale;
  }

  InitBaseConvKernel(param);
  return true;
}

template <>
void ConvBNAddReluKernel<CPU, float>::Compute(
    const FusionConvBNAddReluParam<CPU> &param) {
  switch (param.ExecMode()) {
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S1_FLOAT:
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2_FLOAT:
      DepthwiseConv3x3<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE5x5_FLOAT:
      DepthwiseConv5x5<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<8, 3>(param);
      break;
    case ConvParam<CPU>::EXEC_GEMM_FLOAT:
      GemmConv<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_GEMM1x1s1_FLOAT:
      GemmConv1x1s1<float, float>(param, nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S1_FLOAT:
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S2_FLOAT:
      SlidingwindowConv3x3<float, float>(param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }

  if (param.Bias()->dims() == param.Output()->dims()) {
    math::ScaleAddChannelWise<RELU>(param.Output(), param.InputScale(),
                                    param.InputBias(), param.Bias(),
                                    param.Output());
  } else {
    math::ScaleAddChannelWise<IDENTITY>(param.Output(), param.InputScale(),
                                        param.InputBias(), param.Output());
    math::AddElememtWise<RELU>(param.Output(), param.Bias(), param.Axis(),
                               param.Output());
  }
}

template class ConvBNAddReluKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
