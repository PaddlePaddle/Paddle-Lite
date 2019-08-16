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

#ifdef DROPOUT_OP

#include "operators/kernel/dropout_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool DropoutKernel<GPU_CL, float>::Init(DropoutParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("dropout", "dropout_kernel.cl");
  return true;
}

template <>
void DropoutKernel<GPU_CL, float>::Compute(const DropoutParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  auto *input_image = param.InputX()->GetCLImage();
  auto *output_image = param.Out()->GetCLImage();
  const float dropoutProb = param.DropoutProb();
  const auto &inputDim = param.InputX()->dims();
  int input_dims[4] = {1, 1, 1, 1};
  // 1 1000 1 1
  for (int i = 0; i < inputDim.size(); i++) {
    input_dims[4 - inputDim.size() + i] = inputDim[i];
  }
  int out_W = input_dims[1];
  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(float), &dropoutProb);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
