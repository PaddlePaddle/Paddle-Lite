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

#ifdef LRN_OP

#include "operators/kernel/lrn_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool LrnKernel<GPU_CL, float>::Init(LrnParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("lrn", "lrn_kernel.cl");
  return true;
}

template <>
void LrnKernel<GPU_CL, float>::Compute(const LrnParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Out());

  auto input_image = param.InputX()->GetCLImage();
  auto x_dims = param.InputX()->dims();
  auto output_image = param.Out()->GetCLImage();

  const int N = x_dims[0];
  const int C = x_dims[1];
  const int H = x_dims[2];
  const int W = x_dims[3];

  const int n = param.N();
  const float alpha = param.Alpha();
  const float beta = param.Beta();
  const float k = param.K();
  DLOG << "n=" << n;
  DLOG << "alpha=" << alpha;
  DLOG << "beta=" << beta;
  DLOG << "k=" << k;
  DLOG << default_work_size;
  DLOG << C;
  DLOG << W;
  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &n);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(float), &k);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(float), &alpha);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(float), &beta);

  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
