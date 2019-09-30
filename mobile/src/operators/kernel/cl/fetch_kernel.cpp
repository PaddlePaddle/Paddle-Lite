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
  return true;
}

template <>
void FetchKernel<GPU_CL, float>::Compute(const FetchParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.InputX());

  const int col = param.Col();
  auto input = param.InputX()->GetCLImage();
  auto *out = &param.Out()->at(col);
  out->Resize(param.InputX()->dims());
  out->mutable_data<float>();

  DLOG << "fetch kernel out dims = " << out->dims();
  DLOG << "fetch kernel out memory size = " << out->memory_size();

  auto dim = param.InputX()->dims();
  size_t new_dims[] = {1, 1, 1, 1};

  for (int j = 0; j < dim.size(); ++j) {
    new_dims[4 - dim.size() + j] = dim[j];
  }

  size_t in_ch, in_height, in_width;

  in_ch = new_dims[1];
  in_height = new_dims[2];
  in_width = new_dims[3];
  int size_ch = in_height * in_width;
  int size_block = size_ch * 4;
  int size_batch = size_ch * in_ch;

  framework::CLTensor out_cl_tensor(this->cl_helper_.CLContext(),
                                    this->cl_helper_.CLCommandQueue());
  out_cl_tensor.Resize(out->dims());
  cl_mem outBuffer = out_cl_tensor.mutable_data<float>();

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(int), &in_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(int), &in_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &outBuffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &size_ch);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(int), &size_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(int), &size_batch);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(int), &in_ch);
  CL_CHECK_ERRORS(status);

  //  cl_event wait_event = param.InpdutX()->GetClEvent();
  status =
      clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
                             default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);

  clFinish(this->cl_helper_.CLCommandQueue());

  DLOG << "fetch kernel out dims = " << out->dims();
  DLOG << "fetch kernel out memory size = " << out->memory_size();

  DLOG << "fetch kernel out_cl_tensor dims = " << out_cl_tensor.dims();
  DLOG << "fetch kernel out_cl_tensor memery size = "
       << out_cl_tensor.memory_size();
  memcpy(out->data<float>(), out_cl_tensor.Data<float>(),
         sizeof(float) * out->numel());
}

template class FetchKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
