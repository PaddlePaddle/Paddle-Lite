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

#ifdef BATCHNORM_OP

#include "operators/kernel/batchnorm_kernel.h"
#include <cmath>

namespace paddle_mobile {
namespace operators {

template <>
bool BatchNormKernel<GPU_CL, float>::Init(BatchNormParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("batchnorm", "batchnorm_kernel.cl");
  const framework::CLImage *mean = param->InputMean();
  const framework::CLImage *variance = param->InputVariance();
  const framework::CLImage *scale = param->InputScale();
  const framework::CLImage *bias = param->InputBias();
  const float epsilon = param->Epsilon();

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  const int C = mean->numel();
  float inv_std_ptr[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }
  float *new_scale_ptr = new float[C];
  float *new_bias_ptr = new float[C];

  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
  }

  framework::CLImage *new_scale = new framework::CLImage();
  new_scale->SetTensorData(new_scale_ptr, variance->dims());
  new_scale->InitCLImage(this->cl_helper_.CLContext(),
                         this->cl_helper_.CLCommandQueue());

  framework::CLImage *new_bias = new framework::CLImage();
  new_bias->SetTensorData(new_bias_ptr, variance->dims());
  new_bias->InitCLImage(this->cl_helper_.CLContext(),
                        this->cl_helper_.CLCommandQueue());

  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);

  delete[](new_scale_ptr);
  delete[](new_bias_ptr);

  return true;
}

template <>
void BatchNormKernel<GPU_CL, float>::Compute(
    const BatchNormParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.OutputY());

  auto input = param.InputX()->GetCLImage();
  auto out = param.OutputY()->GetCLImage();
  auto new_scale = param.NewScale()->GetCLImage();
  auto new_bias = param.NewBias()->GetCLImage();
  const int out_width = default_work_size[1];
  DLOG << *param.InputX();
  DLOG << *param.NewBias();
  DLOG << *param.NewScale();
  DLOG << default_work_size[0];
  DLOG << default_work_size[1];
  DLOG << default_work_size[2];
  DLOG << out_width;
  DLOG << *param.OutputY();
  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_int), &out_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &new_scale);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &new_bias);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &out);
  CL_CHECK_ERRORS(status);
  status =
      clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
                             default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

template class BatchNormKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
