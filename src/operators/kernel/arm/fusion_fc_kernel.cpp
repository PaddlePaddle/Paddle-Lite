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

#ifdef FUSION_FC_OP

#include "operators/kernel/fusion_fc_kernel.h"
#include "operators/kernel/central-arm-func/fusion_fc_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FusionFcKernel<CPU, float>::Init(FusionFcParam<CPU> *param) {
  int M = (int)param->InputX()->dims()[0];
  if (M == 1) {
    int r = param->InputY()->dims()[0];
    int c = param->InputY()->dims()[1];
    float *B = param->InputY()->data<float>();
    framework::Tensor matrix_trans;
    float *trans_b = matrix_trans.mutable_data<float>({r, c});
    int index = 0;
    for (int j = 0; j < c; j++) {
      for (int i = 0; i < r; i++) {
        trans_b[index++] = B[i * c + j];
      }
    }
    index = 0;
    for (int j = 0; j < c; j++) {
      for (int i = 0; i < r; i++) {
        B[index] = trans_b[index];
        index++;
      }
    }
  }
  return true;
}

template <>
void FusionFcKernel<CPU, float>::Compute(const FusionFcParam<CPU> &param) {
  FusionFcCompute<float, float>(param);
  param.Out()->set_lod(param.InputX()->lod());
}

template class FusionFcKernel<CPU, float>;

#ifdef FUSION_FC_INT8_OP
template <>
bool FusionFcKernel<CPU, int8_t>::Init(FusionFcParam<CPU> *param) {
  return true;
}

template <>
void FusionFcKernel<CPU, int8_t>::Compute(const FusionFcParam<CPU> &param) {
  FusionFcCompute<int8_t, int32_t>(param);
  param.Out()->set_lod(param.InputX()->lod());
}

template class FusionFcKernel<CPU, int8_t>;
#endif

}  // namespace operators
}  // namespace paddle_mobile

#endif
