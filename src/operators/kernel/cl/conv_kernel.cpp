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

#ifdef CONV_OP

#include "operators/kernel/conv_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvKernel<GPU_CL, float>::Init(ConvParam<GPU_CL> *param) {
  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");

  param->Filter()->InitCLImage(cl_helper_.CLContext(),
                               this->cl_helper_.CLCommandQueue());

  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);
  param->SetOffset(offset);

  DLOG << " init helper: " << &cl_helper_;
  DLOG << " conv kernel add kernel ~ ";
  DLOG << " width of one block: " << param->Filter()->WidthOfOneBlock();
  DLOG << " height of one block: " << param->Filter()->HeightOfOneBlock();
  DLOG << " filter dims: " << param->Filter()->dims();

  if (param->Filter()->WidthOfOneBlock() == 1 &&
      param->Filter()->HeightOfOneBlock() == 1) {
    DLOG << " here1 ";
    this->cl_helper_.AddKernel("conv_1x1", "conv_kernel.cl");

  } else if (param->Filter()->dims()[1] == 1) {
    DLOG << " here2 ";
    this->cl_helper_.AddKernel("depth_conv_3x3", "conv_kernel.cl");

  } else if (param->Filter()->WidthOfOneBlock() == 3 &&
             param->Filter()->HeightOfOneBlock() == 3) {
    DLOG << " here3 ";
    this->cl_helper_.AddKernel("conv_3x3", "conv_kernel.cl");

  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }

  return true;
}

template <>
void ConvKernel<GPU_CL, float>::Compute(const ConvParam<GPU_CL> &param) {
  DLOG << " Compute helper: " << &cl_helper_;
  DLOG << " begin compute ";
  auto kernel = this->cl_helper_.KernelAt(0);
  DLOG << " get work size ";
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Output());
  DLOG << " end work size ";
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  auto input = param.Input()->GetCLImage();

  DLOG << " get Input ";

  auto filter = param.Filter()->GetCLImage();

  DLOG << " get Filter ";

  auto output = param.Output()->GetCLImage();

  DLOG << " get Output ";

  int stride = param.Strides()[0];
  int offset = param.Offset();
  int input_c = param.Input()->CBlock();
  int dilation = param.Dilations()[0];
  int input_width = param.Input()->WidthOfOneBlock();
  int input_height = param.Input()->HeightOfOneBlock();

  int output_width = param.Output()->WidthOfOneBlock();
  int output_height = param.Output()->HeightOfOneBlock();

  cl_int status;

  DLOG << " begin set kernel arg ";
  DLOG << " c block " << c_block;
  DLOG << " w " << w;
  DLOG << " nh " << nh;
  DLOG << " stride " << stride;
  DLOG << " offset " << offset;
  DLOG << " input_c " << input_c;
  DLOG << " dilation " << dilation;
  DLOG << " input width " << input_width;
  DLOG << " input height " << input_height;
  DLOG << " output width " << output_width;
  DLOG << " output height " << output_height;

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

  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 6, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 7, sizeof(int), &offset);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 8, sizeof(int), &input_c);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 9, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 10, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 11, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 12, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, 13, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  DLOG << " end set kernel arg ";

  DLOG << " begin enqueue ";

  status =
      clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
                             default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
  DLOG << " end enqueue ";
}

template class ConvKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
