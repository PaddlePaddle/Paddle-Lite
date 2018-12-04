///* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */
//
//#ifdef DEQUANT_OP
//
//#include "operators/kernel/dequantize_kernel.h"
//
// namespace paddle_mobile {
// namespace operators {
//
// template <>
// bool DequantizeKernel<GPU_CL, float>::Init(DequantizeParam<GPU_CL> *param) {
//  DLOG << " depthwise conv kernel init begin ";
//  PADDLE_MOBILE_ENFORCE(
//      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
//          param->Paddings()[0] == param->Paddings()[1],
//      "need equal");
//  param->Filter()->InitCLImage(cl_helper_.CLContext(),
//                               this->cl_helper_.CLCommandQueue());
//  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
//               static_cast<int>(param->Paddings()[1]);
//  param->SetOffset(offset);
//  this->cl_helper_.AddKernel("depth_conv_3x3", "conv_add_bn_relu_kernel.cl");
//  DLOG << " depthwise conv kernel init end ";
//  return true;
//}
//
// template <>
// void DequantizeKernel<GPU_CL, float>::Compute(
//    const DequantizeParam<GPU_CL> &param) {
//  auto kernel = this->cl_helper_.KernelAt(0);
//  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Output());
//  int c_block = default_work_size[0];
//  int w = default_work_size[1];
//  int nh = default_work_size[2];
//  auto input = param.Input()->GetCLImage();
//  auto filter = param.Filter()->GetCLImage();
//  auto output = param.Output()->GetCLImage();
//  int stride = param.Strides()[0];
//  int offset = param.Offset();
//  int input_c = reinterpret_cast<framework::CLImageConverterFolder *>(
//                    param.Input()->Converter())
//                    ->GetCBlock();
//  int dilation = param.Dilations()[0];
//
//  int input_width = param.Input()->dims()[3];
//  int input_height = param.Input()->dims()[2];
//  int output_width = param.Output()->dims()[3];
//  int output_height = param.Output()->dims()[2];
//
//  cl_int status;
//
//  status = clSetKernelArg(kernel, 0, sizeof(int), &c_block);
//  status = clSetKernelArg(kernel, 1, sizeof(int), &w);
//  status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
//  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
//  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &filter);
//  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
//  status = clSetKernelArg(kernel, 6, sizeof(int), &stride);
//  status = clSetKernelArg(kernel, 7, sizeof(int), &offset);
//  status = clSetKernelArg(kernel, 8, sizeof(int), &input_c);
//  status = clSetKernelArg(kernel, 9, sizeof(int), &dilation);
//  status = clSetKernelArg(kernel, 10, sizeof(int), &input_width);
//  status = clSetKernelArg(kernel, 11, sizeof(int), &input_height);
//  status = clSetKernelArg(kernel, 12, sizeof(int), &output_width);
//  status = clSetKernelArg(kernel, 13, sizeof(int), &output_height);
//
//  CL_CHECK_ERRORS(status);
//
//  //  cl_event out_event = param.Output()->GetClEvent();
//  //  cl_event wait_event = param.Input()->GetClEvent();
//
//  status = clEnqueueNDRangeKernel(
//      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
//      NULL, default_work_size.data(), NULL, 0, NULL, NULL);
//
//  CL_CHECK_ERRORS(status);
//}
//
// template class DepthwiseConvKernel<GPU_CL, float>;
//
//}  // namespace operators
//}  // namespace paddle_mobile
//
//#endif
