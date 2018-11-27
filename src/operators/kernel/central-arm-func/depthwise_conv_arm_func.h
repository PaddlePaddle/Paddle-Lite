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

#ifdef DEPTHWISECONV_OP

#pragma once
#include <vector>
#include "operators/kernel/central-arm-func/conv_arm_func.h"
#include "operators/math/depthwise_conv3x3.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void DepthwiseConvCompute(const ConvParam<CPU> &param) {
  Tensor Bias;
  Bias.mutable_data<float>({param.Groups()});
  if (param.Groups() == param.Input()->dims()[1] &&
      param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
      param.Filter()->dims()[2] == 3 && param.Strides()[0] == 1) {
    math::DepthwiseConv3x3s1p1(param.Input(), param.Filter(), param.Output(),
                               &Bias, false);
  } else if (param.Groups() == param.Input()->dims()[1] &&
             param.Input()->dims()[1] == param.Output()->dims()[1] &&
             param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
             param.Filter()->dims()[2] == 3 && param.Strides()[0] == 2) {
    //    math::DepthwiseConv3x3(param.Input(), param.Strides(),
    //    param.Paddings(),
    //                           param.Filter(), &Bias, param.Output(), false);
    math::DepthwiseConv3x3s2p1v2(param.Input(), param.Filter(), param.Output(),
                                 Bias, false);

  } else {
    GemmConv<float, float>(param);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
