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

#ifdef ELEMENTWISEMUL_OP

#include "operators/kernel/elementwise_mul_kernel.h"
#include "framework/cl/cl_image.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseMulKernel<GPU_CL, float>::Init(
    ElementwiseMulParam<GPU_CL> *param) {
  DLOG << "-----init add-----";
  framework::CLImage *bias = reinterpret_cast<framework::CLImage *>(
      const_cast<framework::CLImage *>(param->InputY()));
  if (bias->dims() == param->InputX()->dims()) {
    this->cl_helper_.AddKernel("elementwise_mul", "elementwise_mul_kernel.cl");
  } else if (bias->dims().size() == 4) {
    this->cl_helper_.AddKernel("channel_mul", "elementwise_mul_kernel.cl");
  } else {
    DLOG << "error:bias dims is error";
  }
  return true;
}

template <>
void ElementwiseMulKernel<GPU_CL, float>::Compute(
    const ElementwiseMulParam<GPU_CL> &param) {
  auto input = param.InputX();
  auto bias = param.InputY();
  auto output = param.Out();
  cl_int status;
  auto kernel = this->cl_helper_.KernelAt(0);
  if (bias->dims() == input->dims()) {
    cl_mem input_image = input->GetCLImage();
    cl_mem bias_image = bias->GetCLImage();
    cl_mem output_image = output->GetCLImage();
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                            reinterpret_cast<void *>(&input_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                            reinterpret_cast<void *>(&bias_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                            reinterpret_cast<void *>(&output_image));
    CL_CHECK_ERRORS(status);
    auto width = input->ImageWidth();
    auto height = input->ImageHeight();
    size_t global_work_size[2] = {width, height};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else if (bias->dims().size() == 4) {
    DLOG << "zp7 444";
    cl_mem input_image = input->GetCLImage();
    cl_mem bias_image = bias->GetCLImage();
    cl_mem output_image = output->GetCLImage();
    int tensor_w = input->dims()[input->dims().size() - 1];
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                            reinterpret_cast<void *>(&input_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                            reinterpret_cast<void *>(&bias_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                            reinterpret_cast<void *>(&output_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 3, sizeof(cl_int),
                            reinterpret_cast<void *>(&tensor_w));
    CL_CHECK_ERRORS(status);
    auto width = input->ImageWidth();
    auto height = input->ImageHeight();
    DLOG << "dede:" << width << "," << height;
    size_t global_work_size[2] = {width, height};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else {
    DLOG << "error:bias dims is error";
  }
}

template class ElementwiseMulKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
