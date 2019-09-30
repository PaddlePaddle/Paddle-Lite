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
#ifdef RELU_OP

#include "operators/kernel/activation_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ReluKernel<GPU_CL, float>::Init(ReluParam<GPU_CL>* param) {
  this->cl_helper_.AddKernel("relu", "relu.cl");
  //  this->cl_helper_.AddKernel("relu_p0", "relu.cl");
  //  this->cl_helper_.AddKernel("relu_p1", "relu.cl");
  //  const auto dim =
  //      const_cast<framework::CLImage*>(param->InputX())->ImageDims();
  //  param->getMidImage().InitEmptyImage(this->cl_helper_.CLContext(),
  //                                      this->cl_helper_.CLCommandQueue(),
  //                                      dim);
  return true;
}

template <>
void ReluKernel<GPU_CL, float>::Compute(const ReluParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  //  auto kernel_p0 = this->cl_helper_.KernelAt(1);
  //  auto kernel_p1 = this->cl_helper_.KernelAt(2);
  const auto* input = param.InputX();
  auto* output = param.Out();
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*output);
  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  //  auto tImage =
  //      const_cast<ReluParam<GPU_CL>&>(param).getMidImage().GetCLImage();
  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);
  //  clSetKernelArg(kernel_p0, 0, sizeof(cl_mem), &inputImage);
  //  clSetKernelArg(kernel_p0, 0, sizeof(cl_mem), &tImage);
  //  clSetKernelArg(kernel_p1, 0, sizeof(cl_mem), &tImage);
  //  clSetKernelArg(kernel_p1, 1, sizeof(cl_mem), &outputImage);
  const size_t work_size[2] = {input->ImageWidth(), input->ImageHeight()};

  //  cl_event out_event = param.Out()->GetClEvent();
  //  cl_event wait_event = param.InputX()->GetClEvent();

  status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                                  NULL, work_size, NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
  //  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel_p1, 3,
  //  NULL,
  //                         work_size, NULL, 0, NULL, NULL);
}

template class ReluKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
