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
#include <framework/cl/cl_half.h>
#include <iostream>
#include "framework/cl/cl_image.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseMulKernel<GPU_CL, float>::Init(
    ElementwiseMulParam<GPU_CL> *param) {
  framework::CLImage *bias = reinterpret_cast<framework::CLImage *>(
      const_cast<framework::CLImage *>(param->InputY()));
  if (bias->dims() == param->InputX()->dims()) {
    DLOG << "init element wise mul";
    this->cl_helper_.AddKernel("elementwise_mul", "elementwise_mul_kernel.cl");
  } else if (bias->dims().size() == 1) {
    DLOG << "init channel_mul";
    this->cl_helper_.AddKernel("channel_mul", "elementwise_mul_kernel.cl");
  } else if (bias->dims().size() == 2) {
    // etc. input  1 72 28 28
    // filter 1 72
    DLOG << "init channel_mul_d2";
    this->cl_helper_.AddKernel("channel_mul_d2", "elementwise_mul_kernel.cl");
  } else {
    PADDLE_MOBILE_ENFORCE(false, "element mul not supported yet");
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
  } else if (bias->dims().size() == 1) {
    DLOG << "channel mul";
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
    size_t global_work_size[2] = {width, height};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else if (bias->dims().size() == 2) {
    DLOG << "channel mul d2";

    // etc. input  1 72 28 28
    // filter 1 72   -->  1 1 1 72
    DLOG << "input->ImageDims():  " << input->ImageDims();
    DLOG << "bias->ImageDims():  " << bias->ImageDims();
    DLOG << "out->ImageDims():  " << output->ImageDims();

    DLOG << "channel mul d2";
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
    size_t global_work_size[2] = {width, height};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);

    //    bias->PrintTensor(*bias);
  } else {
    PADDLE_MOBILE_ENFORCE(false, "element mul not support this situation yet")
  }
}

template class ElementwiseMulKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
