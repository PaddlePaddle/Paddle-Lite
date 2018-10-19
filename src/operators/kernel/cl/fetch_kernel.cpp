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
#include "framework/cl/cl_tensor.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FetchKernel<GPU_CL, float>::Init(FetchParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("fetch", "fetch_kernel.cl");
  auto *out = param->Out();
  out->mutable_data<float>();
  return true;
}

template <>
void FetchKernel<GPU_CL, float>::Compute(const FetchParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.InputX());

  auto input = param.InputX()->GetCLImage();
  auto *out = param.Out();

  const auto &dim = param.InputX()->dims();
  size_t new_dims[] = {1, 1, 1, 1};

  for (int j = 0; j < dim.size(); ++j) {
    new_dims[4 - dim.size() + j] = dim[j];
  }

  size_t N, C, in_height, in_width;

  N = new_dims[0];
  C = new_dims[1];
  in_height = new_dims[2];
  in_width = new_dims[3];

  int size_ch = in_height * in_width;
  int size_block = size_ch * 4;
  int size_batch = size_ch * C;

  CLTensor out_cl_tensor(this->cl_helper_.CLContext(),
                         this->cl_helper_.CLCommandQueue());
  out_cl_tensor.Resize(out->dims());
  cl_mem outBuffer = out_cl_tensor.mutable_data<float>();

  clSetKernelArg(kernel, 0, sizeof(int), &in_height);
  clSetKernelArg(kernel, 1, sizeof(int), &in_width);
  clSetKernelArg(kernel, 2, sizeof(int), &size_ch);
  clSetKernelArg(kernel, 3, sizeof(int), &size_block);
  clSetKernelArg(kernel, 4, sizeof(int), &size_batch);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), &input);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), &outBuffer);

  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
                         default_work_size.data(), NULL, 0, NULL, NULL);

  memcpy(out->data<float>(), out_cl_tensor.Data<float>(), out->memory_size());
}

template class FetchKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
