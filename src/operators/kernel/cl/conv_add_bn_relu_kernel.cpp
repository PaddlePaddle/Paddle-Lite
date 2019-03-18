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

#ifdef FUSION_CONVADDBNRELU_OP

#include "operators/kernel/conv_add_bn_relu_kernel.h"
#include <cmath>
#include "framework/cl/cl_image.h"
#include "framework/cl/cl_tool.h"

namespace paddle_mobile {
namespace operators {
bool optimise = true;
template <>
bool ConvAddBNReluKernel<GPU_CL, float>::Init(
    FusionConvAddBNReluParam<GPU_CL> *param) {
  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");

  param->Bias()->InitCLImage(cl_helper_.CLContext(),
                             cl_helper_.CLCommandQueue());

  //  const CL *mean = param->InputMean();
  const framework::CLImage *mean = param->InputMean();
  const framework::CLImage *variance = param->InputVariance();
  const framework::CLImage *scale = param->InputScale();
  const framework::CLImage *bias = param->InputBias();
  const float epsilon = param->Epsilon();

  const int C = mean->numel();

  //  for (int j = 0; j < C; ++j) {
  //    DLOG << " mean - " << j << mean->data<float>()[j];
  //  }
  //
  //  for (int j = 0; j < C; ++j) {
  //    DLOG << " variance - " << j << variance->data<float>()[j];
  //  }
  //
  //  for (int j = 0; j < C; ++j) {
  //    DLOG << " scale - " << j << scale->data<float>()[j];
  //  }
  //
  //  for (int j = 0; j < C; ++j) {
  //    DLOG << " bias - " << j << bias->data<float>()[j];
  //  }

  //
  //  DLOG << " climage mean: " << *mean;
  //  DLOG << " climage variance: " << *variance;
  //  DLOG << " climage scale: " << *scale;
  //  DLOG << " climage bias: " << *bias;

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  float inv_std_ptr[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }
  float *new_scale_ptr = new float[C];
  float *new_bias_ptr = new float[C];

  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
  }

  framework::CLImage *new_scale = new framework::CLImage();

  //  for (int j = 0; j < C; ++j) {
  //    DLOG << " new scale - " << j << new_scale_ptr[j];
  //  }
  //
  //  for (int j = 0; j < C; ++j) {
  //    DLOG << " new bias - " << j << new_bias_ptr[j];
  //  }

  new_scale->SetTensorData(new_scale_ptr, variance->dims());
  new_scale->InitCLImage(this->cl_helper_.CLContext(),
                         cl_helper_.CLCommandQueue());

  //  DLOG << " climage - y bias: " << *(param->Bias());
  //
  //  DLOG << " climage - new scale: " << *new_scale;

  framework::CLImage *new_bias = new framework::CLImage();

  new_bias->SetTensorData(new_bias_ptr, variance->dims());
  new_bias->InitCLImage(this->cl_helper_.CLContext(),
                        cl_helper_.CLCommandQueue());

  //  DLOG << " climage - new bias: " << *new_bias;
  //
  //  DLOG << " climage - filter: " << *(param->Filter());

  param->SetNewScale(new_scale);
  param->SetNewBias(new_bias);

  delete[](new_scale_ptr);
  delete[](new_bias_ptr);

  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");

  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);

  param->SetOffset(offset);

  /*
  if (param->Filter()->dims()[2] == 1 &&
      param->Filter()->dims()[3] == 1 &&
      (param->Filter()->dims()[0] % 16) == 0) {
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_1x1_4", "conv_add_bn_relu_kernel.cl");
    DLOG << " conv add bn relu conv 1x1 4";
  }
  */
  if (param->Filter()->dims()[2] == 1 && param->Filter()->dims()[3] == 1) {
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());
    if (optimise) {
      this->cl_helper_.AddKernel("conv_1x1_spl", "conv_add_bn_relu_kernel.cl");
    } else {
      this->cl_helper_.AddKernel("conv_1x1", "conv_add_bn_relu_kernel.cl");
    }

    DLOG << " conv add bn relu conv 1x1";
  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] == 3) {
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("depth_conv_3x3", "conv_add_bn_relu_kernel.cl");
    DLOG << " conv add bn relu depth_conv_3x3";

  } else if (param->Filter()->dims()[2] == 3 &&
             param->Filter()->dims()[3] == 3) {
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_3x3", "conv_add_bn_relu_kernel.cl");
    DLOG << " conv add bn relu conv_3x3";
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }

  return true;
}

template <>
void ConvAddBNReluKernel<GPU_CL, float>::Compute(
    const FusionConvAddBNReluParam<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();
  auto biase = param.Bias()->GetCLImage();
  auto new_scale = param.NewScale()->GetCLImage();
  auto new_bias = param.NewBias()->GetCLImage();
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

  //  DLOG << " c block " << c_block;
  //  DLOG << " w " << w;
  //  DLOG << " nh " << nh;
  //  DLOG << " stride " << stride;
  //  DLOG << " offset " << offset;
  //  DLOG << " input_c " << input_c;
  //  DLOG << " dilation " << dilation;
  //  DLOG << " input width " << input_width;
  //  DLOG << " input height " << input_height;
  //  DLOG << " output width " << output_width;
  //  DLOG << " output height " << output_height;
  //  DLOG << " input dim " << param.Input()->dims();
  //  DLOG << " output dim " << param.Output()->dims();
  //  DLOG << " filter dim " << param.Filter()->dims();

  cl_int status;

  if (optimise) {
    if (param.Filter()->dims()[2] == 1 && param.Filter()->dims()[3] == 1) {
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

      status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &new_scale);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &new_bias);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &output);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 9, sizeof(int), &stride);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 10, sizeof(int), &offset);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 11, sizeof(int), &input_c);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 12, sizeof(int), &dilation);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 13, sizeof(int), &input_width);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 14, sizeof(int), &input_height);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 15, sizeof(int), &output_width);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 16, sizeof(int), &output_height);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 17, sizeof(int), &w);
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

      status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &new_scale);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &new_bias);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &output);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 9, sizeof(int), &stride);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 10, sizeof(int), &offset);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 11, sizeof(int), &input_c);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 12, sizeof(int), &dilation);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 13, sizeof(int), &input_width);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 14, sizeof(int), &input_height);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 15, sizeof(int), &output_width);
      CL_CHECK_ERRORS(status);

      status = clSetKernelArg(kernel, 16, sizeof(int), &output_height);
      CL_CHECK_ERRORS(status);

      status = clEnqueueNDRangeKernel(
          this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
          NULL, default_work_size.data(), NULL, 0, NULL, NULL);
      CL_CHECK_ERRORS(status);
    }

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

    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &new_scale);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &new_bias);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &output);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 9, sizeof(int), &stride);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 10, sizeof(int), &offset);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 11, sizeof(int), &input_c);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 12, sizeof(int), &dilation);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 13, sizeof(int), &input_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 14, sizeof(int), &input_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 15, sizeof(int), &output_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, 16, sizeof(int), &output_height);
    CL_CHECK_ERRORS(status);
    status = clEnqueueNDRangeKernel(
        this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  }
}

template class ConvAddBNReluKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
