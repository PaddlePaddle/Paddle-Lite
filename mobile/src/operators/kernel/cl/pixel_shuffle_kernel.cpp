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

#ifdef PIXEL_SHUFFLE_OP

#include "operators/kernel/pixel_shuffle_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool PixelShuffleKernel<GPU_CL, float>::Init(PixelShuffleParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("pixel_shuffle", "pixel_shuffle_kernel.cl");
  return true;
}

template <>
void PixelShuffleKernel<GPU_CL, float>::Compute(
    const PixelShuffleParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Out());

  auto input_image = param.InputX()->GetCLImage();
  auto output_image = param.Out()->GetCLImage();
  auto upscale_factor = param.upscale_factor();

  int input_n = param.InputX()->dims()[0];
  int input_c = param.InputX()->dims()[1];
  int input_h = param.InputX()->dims()[2];
  int input_w = param.InputX()->dims()[3];
  int output_n = param.Out()->dims()[0];
  int output_c = param.Out()->dims()[1];
  int output_h = param.Out()->dims()[2];
  int output_w = param.Out()->dims()[3];

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &input_n);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &input_c);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &input_h);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(int), &input_w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(int), &output_n);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(int), &output_c);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 8, sizeof(int), &output_h);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 9, sizeof(int), &output_w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 10, sizeof(int), &upscale_factor);
  CL_CHECK_ERRORS(status);

  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
