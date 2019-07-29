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

#ifdef LEAKY_RELU_OP

#include <operators/kernel/activation_kernel.h>

namespace paddle_mobile {
namespace operators {
template <>
bool LeakyReluKernel<GPU_CL, float>::Init(
    paddle_mobile::operators::LeakyReluParam<paddle_mobile::GPU_CL> *param) {
  this->cl_helper_.AddKernel("leakyrelu", "leakyrelu_kernel.cl");
  return true;
}

template <>
void LeakyReluKernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::LeakyReluParam<paddle_mobile::GPU_CL>
        &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  auto input = param.InputX();
  cl_mem input_image = input->GetCLImage();
  auto output = param.Out();
  cl_mem out_image = output->GetCLImage();
  float alpha = param.Alpha();
  int out_dims_w = output->dims()[3];

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(float), &alpha);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &out_dims_w);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}
template class LeakyReluKernel<GPU_CL, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
