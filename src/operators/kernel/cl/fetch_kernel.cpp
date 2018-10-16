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

#include "operators/kernel/fetch_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FetchKernel<GPU_CL, float>::Init(FetchParam<GPU_CL> *param) {
  //  this->cl_helper_.AddKernel("fetch", "fetch_kernel.cl");
  return true;
}

template <>
void FetchKernel<GPU_CL, float>::Compute(const FetchParam<GPU_CL> &param) {
  //  auto kernel = this->cl_helper_.KernelAt(0);
  //  auto default_work_size =
  //  this->cl_helper_.DefaultWorkSize(*param.InputX());
  //
  //  auto input = param.InputX()->GetCLImage();
  //  auto *out = param.Out();
  //
  //  const auto &dims = param.InputX()->dims();
  //  const int N = dims[0];
  //  const int C = dims[1];
  //  const int in_height = dims[2];
  //  const int in_width = dims[3];
  //
  //  int size_ch = in_height * in_width;
  //  int size_block = size_ch * 4;
  //  int size_batch = size_ch * C;
  //
  //  // need create outputBuffer
  //  cl_image_format imageFormat;
  //  imageFormat.image_channel_order = CL_RGBA;
  //  imageFormat.image_channel_data_type = CL_FLOAT;
  //  cl_mem outputBuffer;
  //
  //  clSetKernelArg(kernel, 0, sizeof(int), &in_height);
  //  clSetKernelArg(kernel, 1, sizeof(int), &in_width);
  //  clSetKernelArg(kernel, 2, sizeof(int), &size_ch);
  //  clSetKernelArg(kernel, 3, sizeof(int), &size_block);
  //  clSetKernelArg(kernel, 4, sizeof(int), &size_batch);
  //  clSetKernelArg(kernel, 5, sizeof(cl_mem), &input);
  //  clSetKernelArg(kernel, 6, sizeof(cl_mem), &outputBuffer);
  //
  //  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
  //                         default_work_size.data(), NULL, 0, NULL, NULL);
}

template class FetchKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
