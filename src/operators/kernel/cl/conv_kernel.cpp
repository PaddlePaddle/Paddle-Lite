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

  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);
  param->SetOffset(offset);

  DLOG << " init helper: " << &cl_helper_;
  DLOG << " conv kernel add kernel ~ ";
  DLOG << " width of one block: " << param->Filter()->dims()[3];
  DLOG << " height of one block: " << param->Filter()->dims()[2];
  DLOG << " filter dims: " << param->Filter()->dims();

  if (param->Filter()->dims()[2] == 1 && param->Filter()->dims()[3] == 1) {
	  param->Filter()->InitNImage(cl_helper_.CLContext(),
	                              cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_1x1", "conv_kernel.cl");
	  DLOG << "conv 1x1";

  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] == 3) {
	  param->Filter()->InitDWImage(cl_helper_.CLContext(),
	                               cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("depth_conv_3x3", "depthwise_conv_kernel.cl");
	  DLOG << "depth_conv 3x3";

  } else if (param->Filter()->dims()[2] == 3 &&
             param->Filter()->dims()[3] == 3) {
  	param->Filter()->InitCLImage(cl_helper_.CLContext(),
	                                 cl_helper_.CLCommandQueue());
  	this->cl_helper_.AddKernel("conv_3x3", "conv_kernel.cl");
	  DLOG << "conv 3x3";

  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }

  return true;
}

template <>
void ConvKernel<GPU_CL, float>::Compute(const ConvParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();
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
  status = clSetKernelArg(kernel, 1, sizeof(int), &w);
  status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &filter);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
  status = clSetKernelArg(kernel, 6, sizeof(int), &stride);
  status = clSetKernelArg(kernel, 7, sizeof(int), &offset);
  status = clSetKernelArg(kernel, 8, sizeof(int), &input_c);
  status = clSetKernelArg(kernel, 9, sizeof(int), &dilation);
  status = clSetKernelArg(kernel, 10, sizeof(int), &input_width);
  status = clSetKernelArg(kernel, 11, sizeof(int), &input_height);
  status = clSetKernelArg(kernel, 12, sizeof(int), &output_width);
  status = clSetKernelArg(kernel, 13, sizeof(int), &output_height);

  //  cl_event out_event = param.Output()->GetClEvent();
  //  cl_event wait_event = param.Input()->GetClEvent();

  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

template class ConvKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
