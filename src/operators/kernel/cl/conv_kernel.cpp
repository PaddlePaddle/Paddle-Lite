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
bool ConvKernel<GPU_CL, float>::Init(ConvParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("conv_3x3", "conv_kernel.cl");
  return true;
}

template <>
void ConvKernel<GPU_CL, float>::Compute(const ConvParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  size_t global_work_size[3] = {1, 2, 3};
  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL);
}

template class ConvKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
