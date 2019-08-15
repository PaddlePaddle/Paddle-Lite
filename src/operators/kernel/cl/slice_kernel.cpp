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

#ifdef SLICE_OP

#include <framework/cl/cl_tensor.h>
#include <operators/kernel/slice_kernel.h>

namespace paddle_mobile {
namespace operators {
template <>
bool SliceKernel<GPU_CL, float>::Init(
    paddle_mobile::operators::SliceParam<paddle_mobile::GPU_CL> *param) {
  this->cl_helper_.AddKernel("slice", "slice_kernel.cl");
  return true;
}

template <>
void SliceKernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::SliceParam<paddle_mobile::GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.output_);
  auto input = param.input_;
  cl_mem input_image = input->GetCLImage();
  auto output = param.output_;
  cl_mem output_image = output->GetCLImage();
  int starts_0 = param.starts_[0];
  int ends_0 = param.ends_[0];
  int axes_0 = param.axes_[0] - (param.original_output_dims_size_ -
                                 param.output_->dims().size());
  int dims_w = input->dims()[axes_0 + 2];

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &starts_0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &ends_0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &dims_w);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}
template class SliceKernel<GPU_CL, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
