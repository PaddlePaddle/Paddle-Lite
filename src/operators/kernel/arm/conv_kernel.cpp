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

#ifdef CONV_OP

#include "operators/kernel/conv_kernel.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvKernel<CPU, float>::Init(ConvParam<CPU> *param) {
  bool conv3x3 = param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
                 param->Filter()->dims()[2] == 3;
  bool conv5x5 = param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
                 param->Filter()->dims()[2] == 5;
  bool depth3x3 = conv3x3 && param->Groups() == param->Input()->dims()[1] &&
                  param->Input()->dims()[1] == param->Output()->dims()[1];
  bool depth5x5 = conv5x5 && param->Groups() == param->Input()->dims()[1] &&
                  param->Input()->dims()[1] == param->Output()->dims()[1];
  if (param->Filter()->type() == typeid(int8_t)) {
#ifndef __aarch64__
    if (depth3x3 && param->Strides()[0] < 3 &&
        param->Strides()[0] == param->Strides()[1]) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3_INT8;
    } else if (depth5x5 && param->Strides()[0] < 2 &&
               param->Strides()[0] == param->Strides()[1]) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE5x5_INT8;
    } else {
#endif  // __aarch64__
      param->ExecMode() = ConvParam<CPU>::EXEC_GEMM_INT8;
#ifndef __aarch64__
    }
#endif  // __aarch64__
  } else {
    if (depth3x3 && param->Strides()[0] == param->Strides()[1] &&
        param->Strides()[0] == 1 && param->Paddings()[0] == 1 &&
        param->Paddings()[0] == param->Paddings()[1]) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3S1P1_FLOAT;
    } else if (depth3x3 && param->Strides()[0] == param->Strides()[1] &&
               param->Strides()[0] == 2 && param->Paddings()[0] == 0 &&
               param->Paddings()[0] == param->Paddings()[1]) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3S2P0_FLOAT;
    } else if (depth3x3 && param->Strides()[0] == param->Strides()[1] &&
               param->Strides()[0] == 2 && param->Paddings()[0] == 1 &&
               param->Paddings()[0] == param->Paddings()[1]) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3S2P1_FLOAT;
    } else if (depth3x3) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3_FLOAT;
#ifndef __aarch64__
    } else if (depth5x5 && param->Strides()[0] == param->Strides()[1] &&
               param->Strides()[0] == 1) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE5x5_FLOAT;
    } else if (conv3x3 && param->Strides()[0] == param->Strides()[1] &&
               param->Dilations()[0] == param->Dilations()[1] &&
               param->Strides()[0] == 1 && param->Dilations()[0] == 1 &&
               param->Output()->dims()[1] >= 16 &&
               param->Input()->dims()[1] >= 16 &&
               param->Input()->dims()[2] <= 140 /* refered from ncnn */) {
      param->ExecMode() = ConvParam<CPU>::EXEC_WINOGRAD3X3_FLOAT;
      // transform weight
      param->transformed_filter_ = new framework::Tensor;
      operators::math::winograd_transform_weight<8, 3>(
          *param->Filter(), param->transformed_filter_);
#endif
    } else if (conv3x3 && param->Strides()[0] == param->Strides()[1] &&
               param->Dilations()[0] == param->Dilations()[1] &&
               param->Strides()[0] == 1 && param->Dilations()[0] == 1 &&
               param->Input()->dims()[2] >= 48 &&
               param->Output()->dims()[1] <= 24) {
      param->ExecMode() = ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S1_FLOAT;
    } else if (conv3x3 && param->Strides()[0] == param->Strides()[1] &&
               param->Dilations()[0] == param->Dilations()[1] &&
               param->Strides()[0] == 2 && param->Dilations()[0] == 1 &&
               param->Input()->dims()[2] >= 48 &&
               param->Output()->dims()[1] <= 24) {
      param->ExecMode() = ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S2_FLOAT;
    } else {
      param->ExecMode() = ConvParam<CPU>::EXEC_GEMM_FLOAT;
    }
  }
  return true;
}

template <>
void ConvKernel<CPU, float>::Compute(const ConvParam<CPU> &param) {
  switch (param.ExecMode()) {
    case ConvParam<CPU>::EXEC_GEMM_INT8:
      GemmConv<int8_t, int32_t>(param);
      break;
#ifndef __aarch64__
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3_INT8:
      DepthwiseConv3x3<int8_t, int32_t>(param);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE5x5_INT8:
      DepthwiseConv5x5<int8_t, int32_t>(param);
      break;
#endif  // __aarch64__
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S1P1_FLOAT:
      math::DepthwiseConv3x3s1p1(param.Input(), param.Filter(), param.Output(),
                                 nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2P1_FLOAT:
      math::DepthwiseConv3x3s2p1v2(param.Input(), param.Filter(),
                                   param.Output(), nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2P0_FLOAT:
      math::DepthwiseConv3x3s2p0(param.Input(), param.Filter(), param.Output(),
                                 nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3_FLOAT:
      math::DepthwiseConv3x3(param.Input(), param.Strides(), param.Paddings(),
                             param.Filter(), nullptr, param.Output(), false);
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S1_FLOAT:
      math::SlidingwindowConv3x3s1(param.Input(), param.Filter(),
                                   param.Paddings(), param.Output(), nullptr,
                                   false, false);
      break;
    case ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S2_FLOAT:
      math::SlidingwindowConv3x3s2_8channel(param.Input(), param.Filter(),
                                            param.Paddings(), param.Output(),
                                            nullptr, false, false);
      break;
#ifndef __aarch64__
    case ConvParam<CPU>::EXEC_DEPTHWISE5x5_FLOAT:
      DepthwiseConv5x5<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<8, 3>(param);
      break;
#endif  // __aarch64__
    case ConvParam<CPU>::EXEC_GEMM_FLOAT:
      GemmConv<float, float>(param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
}

template class ConvKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
