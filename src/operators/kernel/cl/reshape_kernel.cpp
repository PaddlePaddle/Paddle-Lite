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

#include "operators/kernel/reshape_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ReshapeKernel<GPU_CL, float>::Init(ReshapeParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("reshape", "reshape.cl");
  return true;
}

template <>
void ReshapeKernel<GPU_CL, float>::Compute(const ReshapeParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  const auto *input = param.InputX();
  auto *output = param.Out();
  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
  const auto &inputDim = input->dims();
  const auto &outputDim = output->dims();
  int dims[4] = {inputDim[0], inputDim[1], inputDim[2], inputDim[3]};
  int odims[4] = {outputDim[0], outputDim[1], outputDim[2], outputDim[3]};
  clSetKernelArg(kernel, 2, sizeof(int), dims);
  clSetKernelArg(kernel, 3, sizeof(int), dims + 1);
  clSetKernelArg(kernel, 4, sizeof(int), dims + 2);
  clSetKernelArg(kernel, 5, sizeof(int), dims + 3);
  clSetKernelArg(kernel, 6, sizeof(int), odims);
  clSetKernelArg(kernel, 7, sizeof(int), odims + 1);
  clSetKernelArg(kernel, 8, sizeof(int), odims + 2);
  clSetKernelArg(kernel, 9, sizeof(int), odims + 3);
  const size_t work_size[2] = {output->ImageWidth(), output->ImageHeight()};

  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2, NULL,
                         work_size, NULL, 0, NULL, NULL);
}

template class ReshapeKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
