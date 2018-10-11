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
#include "framework/cl/cl_image.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddBNReluKernel<GPU_CL, float>::Init(
    FusionConvAddBNReluParam<GPU_CL> *param) {
  //  const CL *mean = param->InputMean();
  const framework::CLImage *mean = param->InputMean();

  const framework::CLImage *variance = param->InputVariance();
  const framework::CLImage *scale = param->InputScale();
  const framework::CLImage *bias = param->InputBias();
  const float epsilon = param->Epsilon();

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  const int C = mean->numel();

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

  delete[](new_scale_ptr);
  delete[](new_bias_ptr);

  framework::CLImage *new_scale = new framework::CLImage();

  new_scale->Init(this->cl_helper_.CLContext(), new_scale_ptr,
                  variance->dims());

  framework::CLImage *new_bias = new framework::CLImage();

  new_bias->Init(this->cl_helper_.CLContext(), new_bias_ptr, variance->dims());

  param->SetNewScale(new_scale);

  param->SetNewBias(new_bias);

  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");

  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);

  param->SetOffset(offset);

  if (param->Filter()->WidthOfOneBlock() == 1 &&
      param->Filter()->HeightOfOneBlock() == 1) {
    this->cl_helper_.AddKernel("conv_1x1", "conv_add_bn_relu_kernel.cl");
  } else if (param->Filter()->dims()[1] == 1) {
    this->cl_helper_.AddKernel("depth_conv_3x3", "conv_add_bn_relu_kernel.cl");
  } else if (param->Filter()->WidthOfOneBlock() == 3 &&
             param->Filter()->HeightOfOneBlock() == 3) {
    this->cl_helper_.AddKernel("conv_3x3", "conv_add_bn_relu_kernel.cl");
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
  auto output = param.Output();
  int stride = param.Strides()[0];
  int offset = param.Offset();
  int input_c = param.Input()->CBlock();
  int input_width = param.Input()->WidthOfOneBlock();
  int input_height = param.Input()->HeightOfOneBlock();

  clSetKernelArg(kernel, 0, sizeof(int), &c_block);
  clSetKernelArg(kernel, 1, sizeof(int), &w);
  clSetKernelArg(kernel, 2, sizeof(int), &nh);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &filter);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), &biase);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), &new_scale);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), &new_bias);
  clSetKernelArg(kernel, 8, sizeof(cl_mem), &output);
  clSetKernelArg(kernel, 9, sizeof(int), &stride);
  clSetKernelArg(kernel, 10, sizeof(int), &offset);
  clSetKernelArg(kernel, 11, sizeof(int), &input_c);
  clSetKernelArg(kernel, 12, sizeof(int), &input_width);
  clSetKernelArg(kernel, 13, sizeof(int), &input_height);

  clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
                         default_work_size.data(), NULL, 0, NULL, NULL);
}

template class ConvAddBNReluKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
