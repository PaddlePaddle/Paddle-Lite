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

#ifdef FUSION_CONVADD_OP

#include "operators/kernel/conv_add_kernel.h"

namespace paddle_mobile {
namespace operators {
bool optimise_convadd = true;

template <>
bool ConvAddKernel<GPU_CL, float>::Init(FusionConvAddParam<GPU_CL> *param) {
  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");
  param->Bias()->InitCLImage(cl_helper_.CLContext(),
                             this->cl_helper_.CLCommandQueue());

  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);
  param->SetOffset(offset);

  if (param->Filter()->dims()[2] == 1 && param->Filter()->dims()[3] == 1) {
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());
    if (optimise_convadd) {
      this->cl_helper_.AddKernel("conv_1x1_spl", "conv_add_kernel.cl");
    } else {
      this->cl_helper_.AddKernel("conv_1x1", "conv_add_kernel.cl");
    }
  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] == 3) {
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("depth_conv_3x3", "conv_add_kernel.cl");

  } else if (param->Filter()->dims()[2] == 3 &&
             param->Filter()->dims()[3] == 3) {
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_3x3", "conv_add_kernel.cl");

  } else if (param->Filter()->dims()[2] == 7 &&
             param->Filter()->dims()[3] == 7) {
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_7x7", "conv_add_kernel.cl");
  } else if (param->Filter()->dims()[2] == 5 &&
             param->Filter()->dims()[3] == 5) {
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_5x5", "conv_add_kernel.cl");
  }

  return true;
}

template <>
void ConvAddKernel<GPU_CL, float>::Compute(
    const FusionConvAddParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();
  auto biase = param.Bias()->GetCLImage();
  param.Output()->InitEmptyImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue(),
                                 param.Output()->dims());
  auto output = param.Output()->GetCLImage();
  int stride = param.Strides()[0];
  int offset = param.Offset();
  int input_c = reinterpret_cast<framework::CLImageConverterFolder *>(
                    param.Input()->Converter())
                    ->GetCBlock();
  int dilation = param.Dilations()[0];

  int input_width = param.Input()->dims()[3];
  int input_height = param.Input()->dims()[2];
  int output_width = param.Output()->dims()[3];
  int output_height = param.Output()->dims()[2];

  cl_int status;

  if (optimise_convadd && param.Filter()->dims()[2] == 1 &&
      param.Filter()->dims()[3] == 1) {
    status = clSetKernelArg(kernel, 0, sizeof(int), &c_block);
    CL_CHECK_ERRORS(status);

    int maped_w = maptofactor(w, 4);
    status = clSetKernelArg(kernel, 1, sizeof(int), &maped_w);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &filter);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &biase);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &output);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 7, sizeof(int), &stride);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 8, sizeof(int), &offset);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 9, sizeof(int), &input_c);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 10, sizeof(int), &dilation);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 11, sizeof(int), &input_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 12, sizeof(int), &input_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 13, sizeof(int), &output_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 14, sizeof(int), &output_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 15, sizeof(int), &w);
    CL_CHECK_ERRORS(status);

    const size_t work_size[3] = {
        static_cast<const uint32_t>(default_work_size.data()[0]),
        static_cast<const uint32_t>(maped_w),
        static_cast<const uint32_t>(default_work_size.data()[2])};

    status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel,
                                    default_work_size.size(), NULL, work_size,
                                    NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  } else {
    status = clSetKernelArg(kernel, 0, sizeof(int), &c_block);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 1, sizeof(int), &w);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &filter);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &biase);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &output);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 7, sizeof(int), &stride);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 8, sizeof(int), &offset);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 9, sizeof(int), &input_c);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 10, sizeof(int), &dilation);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 11, sizeof(int), &input_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 12, sizeof(int), &input_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 13, sizeof(int), &output_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 14, sizeof(int), &output_height);
    CL_CHECK_ERRORS(status);

    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  }
}

template class ConvAddKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
