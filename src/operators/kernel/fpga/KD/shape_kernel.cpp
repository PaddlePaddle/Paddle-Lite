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

#ifdef SHAPE_OP

#include "operators/kernel/shape_kernel.h"
#include "operators/kernel/central-arm-func/shape_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void ShapeCompute(const ShapeParam<FPGA>& param) {
  auto* in_t = param.Input();
  auto* out_t = param.Out();
  auto out_data = out_t->mutable_data<int32_t>();
  auto in_dims = in_t->dims();
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = static_cast<int32_t>(in_dims[i]);
  }
}

template <>
bool ShapeKernel<FPGA, float>::Init(ShapeParam<FPGA>* param) {
  param->Out()->mutable_data<int>();
  return true;
}

template <>
void ShapeKernel<FPGA, float>::Compute(const ShapeParam<FPGA>& param) {
  ShapeCompute<float>(param);
}

template class ShapeKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
