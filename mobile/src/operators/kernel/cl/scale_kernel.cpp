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

#ifdef SCALE_OP

#include "operators/kernel/scale_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ScaleKernel<GPU_CL, float>::Init(ScaleParam<GPU_CL>* param) {
  this->cl_helper_.AddKernel("scale", "scale_kernel.cl");
  return true;
}

template <>
void ScaleKernel<GPU_CL, float>::Compute(const ScaleParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  const auto* input = param.InputX();
  auto* output = param.Out();
  const float scale = param.Scale();
  const float bias = param.Bias();
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*output);
  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  int out_width = (output->dims().size() == 4) ? output->dims()[3] : 1;

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(float), &scale);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(float), &bias);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &out_width);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

template class ScaleKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
