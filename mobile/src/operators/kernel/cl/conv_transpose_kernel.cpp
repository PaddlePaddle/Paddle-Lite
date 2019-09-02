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
#ifdef CONV_TRANSPOSE_OP

#include "operators/kernel/conv_transpose_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvTransposeKernel<GPU_CL, float>::Init(
    ConvTransposeParam<GPU_CL>* param) {
  param->Filter()->InitConv2dTransposeFilterCLImage(
      cl_helper_.CLContext(), cl_helper_.CLCommandQueue());
  this->cl_helper_.AddKernel("conv_transpose", "conv_transpose.cl");
  return true;
}

template <>
void ConvTransposeKernel<GPU_CL, float>::Compute(
    const ConvTransposeParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  const auto* input = param.Input();
  auto* output = param.Output();
  auto* filter = param.Filter();
  const int n = input->dims()[0];
  const int input_c = input->dims()[1];
  const int input_c_block = (input_c + 3) / 4;
  const int input_width = input->dims()[3];
  const int input_height = input->dims()[2];
  const int output_c = output->dims()[1];
  const int output_c_block = (output_c + 3) / 4;
  const int output_width = output->dims()[3];
  const int output_height = output->dims()[2];

  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  auto filterImage = filter->GetCLImage();

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(int), &input_c_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &filterImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);

  const size_t work_size[3] = {(size_t)output_c_block, (size_t)input_width,
                               (size_t)(n * input_height)};

  DLOG << "conv transpose " << input_c_block << input_width << input_height
       << output_width << output_height << work_size[0] << work_size[1]
       << work_size[2];

  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
                         work_size, NULL, 0, NULL, NULL);
}

template class ConvTransposeKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
