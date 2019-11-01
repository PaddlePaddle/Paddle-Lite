/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef FLATTEN2_OP

#include "operators/kernel/flatten2_kernel.h"
#include <operators/kernel/reshape_kernel.h>
namespace paddle_mobile {
namespace operators {

template <>
bool Flatten2Kernel<GPU_CL, float>::Init(
    paddle_mobile::operators::FlattenParam<paddle_mobile::GPU_CL> *param) {
  this->cl_helper_.AddKernel("flatten2", "flatten2_kernel.cl");
  return true;
}

template <>
void Flatten2Kernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::FlattenParam<paddle_mobile::GPU_CL>
        &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  const auto *input = param.InputX();
  auto *output = param.Out();
  auto input_image = input->GetCLImage();
  auto output_image = output->GetCLImage();

  int in_width = input->dims()[3];
  int in_height = input->dims()[2];
  int in_c = input->dims()[1];

  int out_width = output->dims()[1];
  DLOG << "flatten2 dims :" << output->dims() << " in: " << input->dims();
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*output);
  DLOG << "flatten2 work size :" << default_work_size.data()[0] << " "
       << default_work_size.data()[1] << "  " << default_work_size.data()[2]
       << "   " << default_work_size.size();

  // const size_t work_size[2] = {output->ImageWidth(), output->ImageHeight()};
  DLOG << "flatten2 work data :" << output->ImageWidth() << "  "
       << output->ImageHeight();

  DLOG << "flatten2 work data 4:" << out_width << " " << in_width << "  "
       << in_height << "   " << in_c;

  int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &out_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &in_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &in_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(int), &in_c);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
