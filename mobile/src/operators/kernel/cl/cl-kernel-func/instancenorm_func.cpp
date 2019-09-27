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

#include "operators/kernel/cl/cl-kernel-func/instancenorm_func.h"
#include <algorithm>
namespace paddle_mobile {
namespace operators {
void InstanceNorm(framework::CLHelper *cl_helper,
                  const InstanceNormParam<GPU_CL> &param) {
  auto kernel = cl_helper->KernelAt(0);

  auto &dims = param.Out()->dims();
  const int n = dims[0];
  const int c_group = (dims[1] + 3) / 4;
  const int h = dims[2];
  const int w = dims[3];
  auto epsilon = param.Epsilon();
  auto input = param.InputX()->GetCLImage();
  auto out = param.Out()->GetCLImage();

  //      DLOG << "Epsilon: " << epsilon;

  auto local_work_size_info = cl_helper->LocalWorkSizeInfo();
  //
  //      DLOG << local_work_size_info.max_work_group_size;
  //      DLOG << local_work_size_info.max_work_item_size0;
  //      DLOG << local_work_size_info.max_work_item_size1;
  //      DLOG << local_work_size_info.max_work_item_size2;
  int maxTotal =
      std::min(static_cast<int>(local_work_size_info.max_work_group_size), 256);
  int local_work_size1 =
      std::min(static_cast<int>(local_work_size_info.max_work_item_size1),
               std::min(256, w));
  int local_work_size2 = 1;
  const size_t work_size[3] = {(size_t)(n * c_group), (size_t)local_work_size1,
                               (size_t)local_work_size2};
  const size_t local_work_size[3] = {(size_t)1, (size_t)local_work_size1,
                                     (size_t)local_work_size2};

  //      DLOG << "work_size" << work_size[0] << " " << work_size[1] << " "
  //           << work_size[2];
  //      DLOG << "local_work_size" << local_work_size[0] << " " <<
  //      local_work_size[1]
  //           << " " << local_work_size[2];
  cl_int status;
  clSetKernelArg(kernel, 0, sizeof(cl_int), &w);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 1, sizeof(cl_int), &h);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 2, sizeof(cl_int), &c_group);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 3, sizeof(cl_int), &local_work_size1);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 4, sizeof(cl_int), &local_work_size2);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 5, sizeof(cl_float), &epsilon);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), &out);
  CL_CHECK_ERRORS(status);
  clEnqueueNDRangeKernel(cl_helper->CLCommandQueue(), kernel, 3, NULL,
                         work_size, local_work_size, 0, NULL, NULL);
}
}  // namespace operators
}  // namespace paddle_mobile
