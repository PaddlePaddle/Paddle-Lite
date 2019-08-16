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

#ifdef SOFTMAX_OP

#include "operators/kernel/softmax_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SoftmaxKernel<GPU_CL, float>::Init(SoftmaxParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("softmax", "softmax.cl");
  return true;
}

template <>
void SoftmaxKernel<GPU_CL, float>::Compute(const SoftmaxParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  const auto *input = param.InputX();
  auto *output = param.Out();
  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  const auto &outputDim = output->dims();

  int dims[4] = {1, 1, 1, 1};

  for (int i = 0; i < outputDim.size(); i++) {
    dims[4 - outputDim.size() + i] = outputDim[i];
  }

  const int out_W = dims[3];

  cl_int status;

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);

  CL_CHECK_ERRORS(status);
}

template class SoftmaxKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
