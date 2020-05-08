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
#ifdef GRID_SAMPLER_OP

#include "operators/kernel/grid_sampler_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool GridSamplerKernel<GPU_CL, float>::Init(GridSamplerParam<GPU_CL>* param) {
  this->cl_helper_.AddKernel("grid_sampler", "grid_sampler_kernel.cl");
  return true;
}

template <>
void GridSamplerKernel<GPU_CL, float>::Compute(
    const GridSamplerParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*(param.Output()));
  cl_int status;
  auto output = param.Output();
  auto input = param.InputX();
  auto grid = param.Grid();
  auto output_image = output->GetCLImage();
  auto input_image = input->GetCLImage();
  auto grid_image = grid->GetCLImage();
  const int out_H = output->dims()[2];
  const int out_W = output->dims()[3];

  status = clSetKernelArg(kernel, 0, sizeof(cl_int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &grid_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);

  const size_t work_size[3] = {default_work_size[0], default_work_size[1],
                               default_work_size[2] / 4};

  status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3,
                                  NULL, work_size, NULL, 0, NULL, NULL);

  CL_CHECK_ERRORS(status);
}

template class GridSamplerKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
