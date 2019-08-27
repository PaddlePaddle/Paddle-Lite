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

#ifdef CONCAT_OP

#include "operators/kernel/concat_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConcatKernel<GPU_CL, float>::Init(ConcatParam<GPU_CL> *param) {
  if (param->Out()->dims().size() < 4) {
    if (param->Out()->dims().size() - param->axis_ == 1) {
      this->cl_helper_.AddKernel("concatByW", "concat_kernel.cl");
    } else {
      this->cl_helper_.AddKernel("concatByH", "concat_kernel.cl");
    }
  } else if (param->Out()->dims().size() >= 4) {
    if (param->Inputs().size() == 2) {
      this->cl_helper_.AddKernel("concatByCWith2Inputs", "concat_kernel.cl");
    } else if (param->Inputs().size() == 3) {
      this->cl_helper_.AddKernel("concatByCWith3Inputs", "concat_kernel.cl");
    } else if (param->Inputs().size() == 4) {
      this->cl_helper_.AddKernel("concatByCWith4Inputs", "concat_kernel.cl");
    } else {
      return false;
    }
  }
  return true;
}

template <>
void ConcatKernel<GPU_CL, float>::Compute(const ConcatParam<GPU_CL> &param) {
  if (param.Out()->dims().size() < 4) {
    auto kernel = this->cl_helper_.KernelAt(0);
    auto inputs = param.Inputs();
    auto *output_image = param.Out()->GetCLImage();
    int out_W = 0;
    if (param.Out()->dims().size() == 3) {
      out_W = param.Out()->dims()[2];
    } else if (param.Out()->dims().size() == 2) {
      out_W = param.Out()->dims()[1];
    }
    int out_H_Start = 0;
    if (param.Out()->dims().size() - param.axis_ == 1) {
      for (int i = 0; i < inputs.size(); i++) {
        int pre_Width = 0;
        for (int k = 0; k < i; ++k) {
          pre_Width += inputs[k]->dims()[inputs[k]->dims().size() - 1];
        }
        int in_w = inputs[i]->dims()[param.Out()->dims().size() - 2];
        auto input_image = inputs[i]->GetCLImage();
        auto default_work_size = this->cl_helper_.DefaultWorkSize(*inputs[i]);
        cl_int status;
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 2, sizeof(int), &in_w);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 3, sizeof(int), &pre_Width);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 4, sizeof(int), &out_W);
        CL_CHECK_ERRORS(status);
        status = clEnqueueNDRangeKernel(
            this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
            NULL, default_work_size.data(), NULL, 0, NULL, NULL);
        CL_CHECK_ERRORS(status);
      }

    } else {
      for (int i = 0; i < inputs.size(); i++) {
        auto input_image = inputs[i]->GetCLImage();
        auto default_work_size = this->cl_helper_.DefaultWorkSize(*inputs[i]);
        cl_int status;
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 2, sizeof(int), &out_W);
        CL_CHECK_ERRORS(status);
        status = clSetKernelArg(kernel, 3, sizeof(int), &out_H_Start);
        CL_CHECK_ERRORS(status);
        status = clEnqueueNDRangeKernel(
            this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
            NULL, default_work_size.data(), NULL, 0, NULL, NULL);
        CL_CHECK_ERRORS(status);
        if (param.Out()->dims().size() == 3) {
          out_H_Start += inputs[i]->dims()[1];
        } else if (param.Out()->dims().size() == 2) {
          out_H_Start += inputs[i]->dims()[0];
        }
      }
    }

  } else {
    auto kernel0 = this->cl_helper_.KernelAt(0);
    auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Out());
    auto inputs = param.Inputs();
    int arg_offset;
    cl_int status;
    if (inputs.size() == 2) {
      auto input_image_0 = inputs[0]->GetCLImage();
      status = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &input_image_0);
      CL_CHECK_ERRORS(status);
      auto input_image_1 = inputs[1]->GetCLImage();
      status = clSetKernelArg(kernel0, 1, sizeof(cl_mem), &input_image_1);
      CL_CHECK_ERRORS(status);
      int C_0 = inputs[0]->dims()[1];
      status = clSetKernelArg(kernel0, 2, sizeof(int), &C_0);
      CL_CHECK_ERRORS(status);
      int C_1 = inputs[1]->dims()[1];
      status = clSetKernelArg(kernel0, 3, sizeof(int), &C_1);
      CL_CHECK_ERRORS(status);
      arg_offset = 4;
    } else if (inputs.size() == 3) {
      auto input_image_0 = inputs[0]->GetCLImage();
      status = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &input_image_0);
      CL_CHECK_ERRORS(status);
      auto input_image_1 = inputs[1]->GetCLImage();
      status = clSetKernelArg(kernel0, 1, sizeof(cl_mem), &input_image_1);
      CL_CHECK_ERRORS(status);
      auto input_image_2 = inputs[2]->GetCLImage();
      status = clSetKernelArg(kernel0, 2, sizeof(cl_mem), &input_image_2);
      CL_CHECK_ERRORS(status);
      int C_0 = inputs[0]->dims()[1];
      status = clSetKernelArg(kernel0, 3, sizeof(int), &C_0);
      CL_CHECK_ERRORS(status);
      int C_1 = inputs[1]->dims()[1];
      status = clSetKernelArg(kernel0, 4, sizeof(int), &C_1);
      CL_CHECK_ERRORS(status);
      int C_2 = inputs[2]->dims()[1];
      status = clSetKernelArg(kernel0, 5, sizeof(int), &C_2);
      CL_CHECK_ERRORS(status);
      arg_offset = 6;
    } else if (inputs.size() == 4) {
      auto input_image_0 = inputs[0]->GetCLImage();
      status = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &input_image_0);
      CL_CHECK_ERRORS(status);
      auto input_image_1 = inputs[1]->GetCLImage();
      status = clSetKernelArg(kernel0, 1, sizeof(cl_mem), &input_image_1);
      CL_CHECK_ERRORS(status);
      auto input_image_2 = inputs[2]->GetCLImage();
      status = clSetKernelArg(kernel0, 2, sizeof(cl_mem), &input_image_2);
      CL_CHECK_ERRORS(status);
      auto input_image_3 = inputs[3]->GetCLImage();
      status = clSetKernelArg(kernel0, 3, sizeof(cl_mem), &input_image_3);
      CL_CHECK_ERRORS(status);
      int C_0 = inputs[0]->dims()[1];
      status = clSetKernelArg(kernel0, 4, sizeof(int), &C_0);
      CL_CHECK_ERRORS(status);
      int C_1 = inputs[1]->dims()[1];
      status = clSetKernelArg(kernel0, 5, sizeof(int), &C_1);
      CL_CHECK_ERRORS(status);
      int C_2 = inputs[2]->dims()[1];
      status = clSetKernelArg(kernel0, 6, sizeof(int), &C_2);
      CL_CHECK_ERRORS(status);
      int C_3 = inputs[3]->dims()[1];
      status = clSetKernelArg(kernel0, 7, sizeof(int), &C_3);
      CL_CHECK_ERRORS(status);
      arg_offset = 8;
    }
    auto *output_image = param.Out()->GetCLImage();
    status =
        clSetKernelArg(kernel0, arg_offset + 0, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    int out_C = param.Out()->dims()[1];
    status = clSetKernelArg(kernel0, arg_offset + 1, sizeof(int), &out_C);
    CL_CHECK_ERRORS(status);
    int out_W = param.Out()->dims()[3];
    status = clSetKernelArg(kernel0, arg_offset + 2, sizeof(int), &out_W);
    CL_CHECK_ERRORS(status);

    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel0, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
