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

#ifdef PAD2D_OP

#include "operators/kernel/pad2d_kernel.h"
#include "framework/cl/cl_tensor.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Pad2DKernel<GPU_CL, float>::Init(Pad2DParam<GPU_CL> *param) {
  DLOG << "Init pad2d";
  this->cl_helper_.AddKernel("pad2d", "pad2d_kernel.cl");
  return true;
}

template <>
void Pad2DKernel<GPU_CL, float>::Compute(const Pad2DParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  cl_int status;
  auto output = param.Out();
  auto input = param.InputX();
  auto output_image = output->GetCLImage();
  auto input_image = input->GetCLImage();
  const int out_H = output->dims()[2];
  const int out_W = output->dims()[3];
  const int input_H = input->dims()[2];
  const int input_W = input->dims()[3];
  const auto &paddings = param.paddings_;
  const int pad_top = paddings[0];
  const int pad_bottom = paddings[1];
  const int pad_left = paddings[2];
  const int pad_right = paddings[3];
  const float pad_value = param.pad_value_;
  const auto &modeStr = param.mode_;
  int mode = 0;
  if (modeStr == "reflect") {
    mode = 1;
  } else if (modeStr == "edge") {
    mode = 2;
  }
  DLOG << "input_H: " << input_H;
  status = clSetKernelArg(kernel, 0, sizeof(cl_int), &input_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_int), &input_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_int), &pad_top);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_int), &pad_bottom);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_int), &pad_left);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_int), &pad_right);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 8, sizeof(cl_int), &mode);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 9, sizeof(cl_float), &pad_value);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 10, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 11, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);

  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);

  CL_CHECK_ERRORS(status);
}

template class Pad2DKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PAD2D_OP
