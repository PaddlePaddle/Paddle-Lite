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

#ifdef ELEMENTWISESUB_OP

#include "operators/kernel/elementwise_sub_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseSubKernel<GPU_CL, float>::Init(
    ElementwiseSubParam<GPU_CL> *param) {
  framework::CLImage *bias = reinterpret_cast<framework::CLImage *>(
      const_cast<framework::CLImage *>(param->InputY()));
  if (bias->dims().size() == 4) {
    if (!bias->isInit()) {
      bias->InitNormalCLImage(cl_helper_.CLContext(),
                              this->cl_helper_.CLCommandQueue());
    }
    DLOG << " bias: " << *bias;
    this->cl_helper_.AddKernel("elementwise_sub", "elementwise_sub_kernel.cl");
  } else {
    DLOG << "error:bias dims not support";
  }
  return true;
}

template <>
void ElementwiseSubKernel<GPU_CL, float>::Compute(
    const ElementwiseSubParam<GPU_CL> &param) {
  auto input = param.InputX();
  auto bias = param.InputY();
  auto output = param.Out();
  cl_int status;
  auto kernel = this->cl_helper_.KernelAt(0);
  if (bias->dims().size() == 4) {
    cl_mem input_image = input->GetCLImage();
    cl_mem bias_image = bias->GetCLImage();
    cl_mem output_image = output->GetCLImage();
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bias_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    auto width = input->ImageWidth();
    auto height = input->ImageHeight();
    size_t global_work_size[2] = {width, height};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else {
    DLOG << "error:bias dims not support";
  }
}

template class ElementwiseSubKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
