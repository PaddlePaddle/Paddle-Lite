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

#ifdef GRU_OP

#include "operators/kernel/gru_kernel.h"
#include "operators/kernel/central-arm-func/gru_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool GruKernel<CPU, float>::Init(GruParam<CPU> *param) {
  return true;
}

template <>
void GruKernel<CPU, float>::Compute(const GruParam<CPU> &param) const {
  auto lod_size = param.InputInput()->lod().size();
  PADDLE_MOBILE_ENFORCE((lod_size == 1),
                        "Current LoD only supports one dimension.");
  auto input_dims = param.InputInput()->dims();
  auto weight_dims = param.InputWeight()->dims();
  int input_size = input_dims[1];
  int frame_size = weight_dims[0];
  PADDLE_MOBILE_ENFORCE(
      (input_size == frame_size * 3),
      "The input_size must be 3 times of frame_size in GRUOp.");
  PADDLE_MOBILE_ENFORCE(
      (weight_dims[1] == frame_size * 3),
      "The shape of Weight matrix must be [frame_size, frame_size * 3].");
  if (param.InputH0()) {
    auto h0_dims = param.InputH0()->dims();
    PADDLE_MOBILE_ENFORCE((h0_dims[1] == frame_size),
                          "The width of H0 must be equal to frame_size.");
  }
  if (param.InputBias()) {
    auto bias_dims = param.InputBias()->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    PADDLE_MOBILE_ENFORCE((bias_height == 1),
                          "The shape of Bias must be [1, frame_size * 3].");
    PADDLE_MOBILE_ENFORCE((bias_width == frame_size * 3),
                          "The shape of Bias must be [1, frame_size * 3].");
  }
  GruCompute<float>(param);
  param.OutHidden()->set_lod(param.InputInput()->lod());
}

template class GruKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
