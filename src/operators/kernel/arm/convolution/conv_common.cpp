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

#include "operators/kernel/arm/convolution/conv_common.h"
#include "operators/math/slidingwindow_utils.h"
#include "operators/math/winograd/winograd_transform.h"

namespace paddle_mobile {
namespace operators {

void InitBaseConvKernel(ConvParam<CPU> *param) {
  bool conv3x3 = param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
                 param->Filter()->dims()[2] == 3;
  bool conv5x5 = param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
                 param->Filter()->dims()[2] == 5;
  bool depth3x3 = conv3x3 && param->Groups() == param->Input()->dims()[1] &&
                  param->Input()->dims()[1] == param->Output()->dims()[1];

  bool depth5x5 = conv5x5 && param->Groups() == param->Input()->dims()[1] &&
                  param->Input()->dims()[1] == param->Output()->dims()[1];

  if (param->Filter()->type() == type_id<int8_t>().hash_code()) {
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
        param->Strides()[0] == 1) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3S1_FLOAT;
    } else if (depth3x3 && param->Strides()[0] == param->Strides()[1] &&
               param->Strides()[0] == 2) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE3x3S2_FLOAT;
    } else if (depth5x5 && param->Strides()[0] == param->Strides()[1] &&
               param->Strides()[0] == 1) {
      param->ExecMode() = ConvParam<CPU>::EXEC_DEPTHWISE5x5_FLOAT;
    } else if (conv3x3 && param->Groups() == 1 &&
               param->Strides()[0] == param->Strides()[1] &&
               param->Dilations()[0] == param->Dilations()[1] &&
               param->Strides()[0] == 1 && param->Dilations()[0] == 1) {
      // transform weight
      Variable *transformed_var = param->GetScope()->Var();
      param->transformed_filter_ =
          transformed_var->GetMutable<framework::LoDTensor>();
      if (param->Input()->dims()[1] >= 32 && param->Output()->dims()[1] >= 32 &&
          param->Output()->dims()[2] > 16 && param->Output()->dims()[3] > 16) {
        math::winograd_transform_weight<8, 3>(*param->Filter(),
                                              param->transformed_filter_);
        param->ExecMode() = ConvParam<CPU>::EXEC_WINOGRAD3X3_FLOAT;
      } else {
        math::slidingwindow_transform_weight<float>(*param->Filter(),
                                                    param->transformed_filter_);
        param->ExecMode() = ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S1_FLOAT;
      }
    } else if (conv3x3 && param->Groups() == 1 &&
               param->Strides()[0] == param->Strides()[1] &&
               param->Dilations()[0] == param->Dilations()[1] &&
               param->Strides()[0] == 2 && param->Dilations()[0] == 1) {
      // transform weight
      Variable *transformed_var = param->GetScope()->Var();
      param->transformed_filter_ =
          transformed_var->GetMutable<framework::LoDTensor>();
      math::slidingwindow_transform_weight<float>(*param->Filter(),
                                                  param->transformed_filter_);
      param->ExecMode() = ConvParam<CPU>::EXEC_SLIDINGWINDOW3x3S2_FLOAT;
    } else {
      param->ExecMode() = ConvParam<CPU>::EXEC_GEMM_FLOAT;
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile
