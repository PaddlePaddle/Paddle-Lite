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

#ifdef ELEMENTWISEADD_OP

#include "operators/kernel/elementwise_add_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseAddKernel<GPU_CL, float>::Init(
    ElementwiseAddParam<GPU_CL> *param) {
  DLOG << "-----init add-----";
  CLImage *bias =
      reinterpret_cast<CLImage *>(const_cast<CLImage *>(param->InputY()));
  if (bias->dims().size() == 4) {
    if (!bias->isInit()) {
      bias->InitNormalCLImage(cl_helper_.CLContext(),
                              this->cl_helper_.CLCommandQueue());
    }
    DLOG << " bias: " << *bias;
    this->cl_helper_.AddKernel("elementwise_add", "elementwise_add_kernel.cl");
  } else if (param->InputY()->dims().size() == 1) {
    if (param->Axis() == param->InputX()->dims().size() - 1) {
      if (!bias->isInit()) {
        bias->InitNormalCLImage(cl_helper_.CLContext(),
                                this->cl_helper_.CLCommandQueue());
      }
      DLOG << " bias: " << *bias;
      this->cl_helper_.AddKernel("width_add", "channel_add_kernel.cl");
    } else if (param->Axis() == param->InputX()->dims().size() - 3) {
      if (!bias->isInit()) {
        bias->InitCLImage(cl_helper_.CLContext(),
                          this->cl_helper_.CLCommandQueue());
      }
      DLOG << " bias: " << *bias;
      this->cl_helper_.AddKernel("channel_add", "channel_add_kernel.cl");
    } else {
      DLOG << "error:bias dims is error";
    }
  } else {
    DLOG << "error:bias dims is error";
  }
  return true;
}

template <>
void ElementwiseAddKernel<GPU_CL, float>::Compute(
    const ElementwiseAddParam<GPU_CL> &param) {
  auto input = param.InputX();
  auto bias = param.InputY();
  auto output = param.Out();
  cl_int status;
  auto kernel = this->cl_helper_.KernelAt(0);
  if (bias->dims().size() == 4) {
    cl_mem input_image = input->GetCLImage();
    cl_mem bias_image = bias->GetCLImage();
    cl_mem output_image = output->GetCLImage();
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                            reinterpret_cast<void *>(&input_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                            reinterpret_cast<void *>(&bias_image));
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                            reinterpret_cast<void *>(&output_image));
    CL_CHECK_ERRORS(status);
    int width = input->ImageWidth();
    int height = input->ImageHeight();
    size_t global_work_size[2] = {width, height};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else if (bias->dims().size() == 1) {
    if (param.Axis() == param.InputX()->dims().size() - 1 ||
        param.Axis() == param.InputX()->dims().size() - 3) {
      cl_mem input_image = input->GetCLImage();
      cl_mem bias_image = bias->GetCLImage();
      cl_mem output_image = output->GetCLImage();
      int tensor_w = input->dims()[input->dims().size() - 1];
      status = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                              reinterpret_cast<void *>(&input_image));
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                              reinterpret_cast<void *>(&bias_image));
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                              reinterpret_cast<void *>(&output_image));
      CL_CHECK_ERRORS(status);
      status = clSetKernelArg(kernel, 3, sizeof(cl_int),
                              reinterpret_cast<void *>(&tensor_w));
      CL_CHECK_ERRORS(status);
      int width = input->ImageWidth();
      int height = input->ImageHeight();
      DLOG << "dede:" << width << "," << height;
      size_t global_work_size[2] = {width, height};
      cl_event out_event = param.Out()->GetClEvent();
      cl_event wait_event = param.InputX()->GetClEvent();
      status =
          clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                                 NULL, global_work_size, NULL, 0, NULL, NULL);
      CL_CHECK_ERRORS(status);
    } else {
      DLOG << "error:bias dims is error";
    }
  } else {
    DLOG << "error:bias dims is error";
  }
}

template class ElementwiseAddKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
