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

#ifdef NEAREST_INTERP_OP

#include <operators/kernel/nearest_interp_kernel.h>

namespace paddle_mobile {
namespace operators {
template <>
bool NearestInterpolationKernel<GPU_CL, float>::Init(
    paddle_mobile::operators::NearestInterpolationParam<paddle_mobile::GPU_CL>
        *param) {
  this->cl_helper_.AddKernel("nearest_interp", "nearest_interp_kernel.cl");
  return true;
}

template <>
void NearestInterpolationKernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::NearestInterpolationParam<
        paddle_mobile::GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  auto input = param.InputX();
  cl_mem input_image = input->GetCLImage();
  auto output = param.Out();
  cl_mem output_image = output->GetCLImage();
  float scale_h = output->dims()[2] / input->dims()[2];
  float scale_w = output->dims()[3] / input->dims()[3];
  int in_dims_h = input->dims()[2];
  int out_dims_h = output->dims()[2];
  int in_dims_w = input->dims()[3];
  int out_dims_w = output->dims()[3];

  cl_int status;

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 2, sizeof(float), &scale_h);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 3, sizeof(float), &scale_w);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 4, sizeof(int), &in_dims_h);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 5, sizeof(int), &out_dims_h);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 6, sizeof(int), &in_dims_w);
  CL_CHECK_ERRORS(status)
  status = clSetKernelArg(kernel, 7, sizeof(int), &out_dims_w);
  CL_CHECK_ERRORS(status)
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status)
}
template class NearestInterpolationKernel<GPU_CL, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
