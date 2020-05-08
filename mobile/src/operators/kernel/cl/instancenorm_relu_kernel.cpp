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

#ifdef FUSION_INSTANCENORM_RELU_OP

#include "operators/kernel/instancenorm_relu_kernel.h"
#include <cmath>
#include "operators/kernel/cl/cl-kernel-func/instancenorm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool InstanceNormReluKernel<GPU_CL, float>::Init(
    FusionInstanceNormReluParam<GPU_CL> *param) {
  auto &dims = param->Out()->dims();
  const int h = dims[2];
  std::string build_options = " -DRELU";
  if (h == 128) {
    build_options += " -DLOCAL_MEM_128";
  } else if (h == 64) {
    build_options += " -DLOCAL_MEM_64";
  }
  this->cl_helper_.AddKernel("instancenorm", "instancenorm_kernel.cl",
                             build_options);
  return true;
}

template <>
void InstanceNormReluKernel<GPU_CL, float>::Compute(
    const FusionInstanceNormReluParam<GPU_CL> &param) {
  InstanceNorm(&this->cl_helper_, param.InputX(), param.Out(), param.Epsilon());
}

template class InstanceNormReluKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
