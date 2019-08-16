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
#ifdef SIGMOID_OP

#include "operators/kernel/activation_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SigmoidKernel<GPU_CL, float>::Init(SigmoidParam<GPU_CL>* param) {
  this->cl_helper_.AddKernel("sigmoid", "sigmoid.cl");
  return true;
}

template <>
void SigmoidKernel<GPU_CL, float>::Compute(const SigmoidParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  const auto* input = param.InputX();
  auto* output = param.Out();
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*output);
  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);
  const size_t work_size[2] = {input->ImageWidth(), input->ImageHeight()};
  status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                                  NULL, work_size, NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

template class SigmoidKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
