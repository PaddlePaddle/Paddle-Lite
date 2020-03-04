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

#ifdef BILINEAR_INTERP_OP

#include <operators/kernel/bilinear_interp_kernel.h>

namespace paddle_mobile {
namespace operators {
template <>
bool BilinearInterpKernel<GPU_CL, float>::Init(
    paddle_mobile::operators::BilinearInterpParam<paddle_mobile::GPU_CL>
        *param) {
  this->cl_helper_.AddKernel("bilinear_interp", "bilinear_interp_kernel.cl");
  return true;
}

template <>
void BilinearInterpKernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::BilinearInterpParam<paddle_mobile::GPU_CL>
        &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  auto input = param.InputX();
  cl_mem input_image = input->GetCLImage();
  auto output = param.Out();
  cl_mem output_image = output->GetCLImage();
  float scale_h, scale_w;
  if (param.AlignCorners()) {
    scale_h = (input->dims()[2] - 1.0f) / (output->dims()[2] - 1.0f);
    scale_w = (input->dims()[3] - 1.0f) / (output->dims()[3] - 1.0f);
  } else {
    scale_h = input->dims()[2] / static_cast<float>(output->dims()[2]);
    scale_w = input->dims()[3] / static_cast<float>(output->dims()[3]);
  }
  float align_delta = 0.0f;
  if (!param.AlignCorners() && param.AlignMode() == 0) {
    align_delta = 0.5f;
  }
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
  status = clSetKernelArg(kernel, 8, sizeof(float), &align_delta);
  CL_CHECK_ERRORS(status)
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status)
}
template class BilinearInterpKernel<GPU_CL, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
