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

#include "operators/kernel/feed_kernel.h"
#include "framework/cl/cl_tensor.h"
namespace paddle_mobile {
namespace operators {

template <>
bool FeedKernel<GPU_CL, float>::Init(FeedParam<GPU_CL> *param) {
  DLOG << "Init feed";
  this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
  return true;
}

template <>
void FeedKernel<GPU_CL, float>::Compute(const FeedParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  cl_int status;
  auto output = param.Out();
  const Tensor *input = param.InputX();
  //  DLOG << *input;
  const float *input_data = input->data<float>();
  int numel = input->numel();
  cl_mem cl_image = output->GetCLImage();
<<<<<<< HEAD
=======
  int c = input->dims()[1];
>>>>>>> upstream/develop
  int height = output->dims()[2];
  int width = output->dims()[3];
  CLTensor input_cl_tensor(this->cl_helper_.CLContext(),
                           this->cl_helper_.CLCommandQueue());
  input_cl_tensor.Resize(input->dims());
  cl_mem inputBuffer = input_cl_tensor.mutable_with_data<float>(input_data);

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), &width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_int), &height);
  CL_CHECK_ERRORS(status);
<<<<<<< HEAD
=======
  status = clSetKernelArg(kernel, 4, sizeof(cl_int), &c);
  CL_CHECK_ERRORS(status);
>>>>>>> upstream/develop

  size_t global_work_size[2] = {width, height};

  //  cl_event out_event = param.Out()->GetClEvent();

  status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                                  NULL, global_work_size, NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

template class FeedKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
