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
  if (this->pre_post_type_ == UINT8_255) {
    this->cl_helper_.AddKernel("feed_with_pre", "feed_kernel.cl");
  } else {
    this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
  }
  return true;
}

template <>
void FeedKernel<GPU_CL, float>::Compute(const FeedParam<GPU_CL> &param) {
  const int col = param.Col();
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Out()));
  cl_int status;
  auto output = param.Out();
  const Tensor *input = &param.InputX()->at(col);
  //  DLOG << *input;

  int numel = input->numel();
  cl_mem output_image = output->GetCLImage();
  const int out_C = output->dims()[1];
  const int out_H = output->dims()[2];
  const int out_W = output->dims()[3];
  const int Stride2 = out_C * out_H * out_W;
  const int Stride1 = out_H * out_W;
  const int Stride0 = out_W;
  framework::CLTensor input_cl_tensor(this->cl_helper_.CLContext(),
                                      this->cl_helper_.CLCommandQueue());
  input_cl_tensor.Resize(input->dims());
  cl_mem inputBuffer;
  if (this->pre_post_type_ == UINT8_255) {
    inputBuffer =
        input_cl_tensor.mutable_with_data<uint8_t>(input->data<uint8_t>());
  } else {
    inputBuffer =
        input_cl_tensor.mutable_with_data<float>(input->data<float>());
  }

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
  CL_CHECK_ERRORS(status);

  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);

  CL_CHECK_ERRORS(status);
}

template class FeedKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
