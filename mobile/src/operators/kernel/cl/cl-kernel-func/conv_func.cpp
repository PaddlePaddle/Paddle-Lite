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

#include "operators/kernel/cl/cl-kernel-func/conv_func.h"
#include <vector>
#include "framework/cl/cl_image_converter.h"
#include "framework/cl/cl_tensor.h"

namespace paddle_mobile {
namespace operators {
bool use_lws = true;
int preferred_lws = 0;
int preferred_lws_divisor = 2;

template <>
void winograd_transform_weight<4, 3>(framework::CLHelper *cl_helper,
                                     framework::CLImage *weight) {}

template <>
void WinogradConv3x3<4, 3>(framework::CLHelper *cl_helper,
                           const ConvParam<GPU_CL> &param, bool ifRelu,
                           const framework::CLImage *biase,
                           const framework::CLImage *new_scale,
                           const framework::CLImage *new_bias) {}

void ConvAddBnReluPt1x2(framework::CLHelper *cl_helper,
                        const ConvParam<GPU_CL> &param, bool ifRelu,
                        const framework::CLImage *biase,
                        const framework::CLImage *new_scale,
                        const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
  default_work_size[1] = (default_work_size[1] + 1) / 2;
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
  int output_c = param.Output()->dims()[1];
  int filter_channel = param.Filter()->dims()[1];
  int input_channel = param.Input()->dims()[1];
  //
  //    DLOG << " c block " << c_block;
  //    DLOG << " w " << w;
  //    DLOG << " nh " << nh;
  //    DLOG << " stride " << stride;
  //    DLOG << " offset " << offset;
  //    DLOG << " input_c " << input_c;
  //    DLOG << " dilation " << dilation;
  //    DLOG << " input width " << input_width;
  //    DLOG << " input height " << input_height;
  //    DLOG << " output width " << output_width;
  //    DLOG << " output height " << output_height;
  //    DLOG << " input dim " << param.Input()->dims();
  //    DLOG << " output dim " << param.Output()->dims();
  //    DLOG << " filter dim " << param.Filter()->dims();

  cl_int status;
  int index = 0;

  status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &w);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &nh);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  if (biase) {
    auto bias_mem = biase->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
    CL_CHECK_ERRORS(status);
  }

  if (new_scale && new_bias) {
    auto new_scale_mem = new_scale->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
    CL_CHECK_ERRORS(status);

    auto new_bias_mem = new_bias->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
    CL_CHECK_ERRORS(status);
  }

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &offset);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_c);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  if (param.Filter()->dims()[2] == 3 && param.Filter()->dims()[3] == 3) {
    if (filter_channel != input_channel) {
      if (filter_channel != 1) {
        status = clSetKernelArg(kernel, index++, sizeof(int), &filter_channel);
        CL_CHECK_ERRORS(status);
        int has_group = 1;
        status = clSetKernelArg(kernel, index++, sizeof(int), &has_group);
        CL_CHECK_ERRORS(status);
      }
    } else {
      status = clSetKernelArg(kernel, index++, sizeof(int), &filter_channel);
      CL_CHECK_ERRORS(status);
      int has_group = 0;
      status = clSetKernelArg(kernel, index++, sizeof(int), &has_group);
      CL_CHECK_ERRORS(status);
    }
  }
  //  DLOG<<"default_work_size"<<default_work_size[0]<<"
  //  "<<default_work_size[1]<<" "<<default_work_size[2];
  auto kernel_work_size = cl_helper->KernelWorkSize(kernel);
  auto tmp0 = default_work_size.data()[0];
  auto tmp1 = default_work_size.data()[1];
  auto tmp2 = default_work_size.data()[2];
  int max_work_size = static_cast<const uint32_t>(kernel_work_size);
  if (preferred_lws_divisor > 1) {
    max_work_size /= preferred_lws_divisor;
  }
  if (preferred_lws > 0 && preferred_lws <= max_work_size) {
    max_work_size = preferred_lws;
  }
  while (tmp1 > max_work_size && max_work_size > 0) {
    tmp1 = tmp1 % 2 == 0 ? tmp1 / 2 : 1;
  }
  while (tmp2 * tmp1 > max_work_size && max_work_size > 0) {
    tmp2 = tmp2 % 2 == 0 ? tmp2 / 2 : 1;
  }
  while (tmp0 * tmp1 * tmp2 > max_work_size && max_work_size > 0) {
    tmp0 = tmp0 % 2 == 0 ? tmp0 / 2 : 1;
  }
  const size_t local_work_size[3] = {static_cast<const uint32_t>(tmp0),
                                     static_cast<const uint32_t>(tmp1),
                                     static_cast<const uint32_t>(tmp2)};
  if (max_work_size > 0 && use_lws) {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), local_work_size, 0, NULL, NULL);
  } else {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
  }
  CL_CHECK_ERRORS(status);
}

void ConvAddBnRelu(framework::CLHelper *cl_helper,
                   const ConvParam<GPU_CL> &param, bool ifRelu,
                   const framework::CLImage *biase,
                   const framework::CLImage *new_scale,
                   const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
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
  int output_c = param.Output()->dims()[1];
  int filter_channel = param.Filter()->dims()[1];
  int input_channel = param.Input()->dims()[1];

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
  int index = 0;

  if (param.Filter()->dims()[2] == 1 && param.Filter()->dims()[3] == 1) {
    status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
    CL_CHECK_ERRORS(status);

    int maped_w = maptofactor(w, 4);
    status = clSetKernelArg(kernel, index++, sizeof(int), &maped_w);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &nh);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
    CL_CHECK_ERRORS(status);

    if (biase) {
      auto bias_mem = biase->GetCLImage();
      status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
      CL_CHECK_ERRORS(status);
    }

    if (new_scale && new_bias) {
      auto new_scale_mem = new_scale->GetCLImage();
      status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
      CL_CHECK_ERRORS(status);

      auto new_bias_mem = new_bias->GetCLImage();
      status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
      CL_CHECK_ERRORS(status);
    }

    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &offset);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &input_c);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &w);
    CL_CHECK_ERRORS(status);

    const size_t work_size[3] = {
        static_cast<const uint32_t>(default_work_size.data()[0]),
        static_cast<const uint32_t>(maped_w),
        static_cast<const uint32_t>(default_work_size.data()[2])};

    auto kernel_work_size = cl_helper->KernelWorkSize(kernel);
    auto tmp0 = work_size[0];
    auto tmp1 = work_size[1];
    auto tmp2 = work_size[2];
    int max_work_size = static_cast<const uint32_t>(kernel_work_size);
    if (preferred_lws_divisor > 1) {
      max_work_size /= preferred_lws_divisor;
    }
    if (preferred_lws > 0 && preferred_lws <= max_work_size) {
      max_work_size = preferred_lws;
    }
    while (tmp1 > max_work_size && max_work_size > 0) {
      tmp1 = tmp1 % 2 == 0 ? tmp1 / 2 : 1;
    }
    while (tmp2 * tmp1 > max_work_size && max_work_size > 0) {
      tmp2 = tmp2 % 2 == 0 ? tmp2 / 2 : 1;
    }
    while (tmp0 * tmp1 * tmp2 > max_work_size && max_work_size > 0) {
      tmp0 = tmp0 % 2 == 0 ? tmp0 / 2 : 1;
    }
    const size_t local_work_size[3] = {static_cast<const uint32_t>(tmp0),
                                       static_cast<const uint32_t>(tmp1),
                                       static_cast<const uint32_t>(tmp2)};
    if (max_work_size > 0 && use_lws) {
      status = clEnqueueNDRangeKernel(cl_helper->CLCommandQueue(), kernel,
                                      default_work_size.size(), NULL, work_size,
                                      local_work_size, 0, NULL, NULL);
    } else {
      status = clEnqueueNDRangeKernel(cl_helper->CLCommandQueue(), kernel,
                                      default_work_size.size(), NULL, work_size,
                                      NULL, 0, NULL, NULL);
    }
    CL_CHECK_ERRORS(status);
  } else {
    status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &w);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &nh);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
    CL_CHECK_ERRORS(status);

    if (biase) {
      auto bias_mem = biase->GetCLImage();
      status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
      CL_CHECK_ERRORS(status);
    }

    if (new_scale && new_bias) {
      auto new_scale_mem = new_scale->GetCLImage();
      status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
      CL_CHECK_ERRORS(status);

      auto new_bias_mem = new_bias->GetCLImage();
      status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
      CL_CHECK_ERRORS(status);
    }

    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &offset);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &input_c);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
    CL_CHECK_ERRORS(status);

    status = clSetKernelArg(kernel, index++, sizeof(int), &output_c);
    CL_CHECK_ERRORS(status);

    if (param.Filter()->dims()[2] == 3 && param.Filter()->dims()[3] == 3) {
      if (filter_channel != input_channel) {
        status = clSetKernelArg(kernel, index++, sizeof(int), &filter_channel);
        CL_CHECK_ERRORS(status);
        int group = input_channel / filter_channel;
        status = clSetKernelArg(kernel, index++, sizeof(int), &group);
        CL_CHECK_ERRORS(status);
      } else {
        status = clSetKernelArg(kernel, index++, sizeof(int), &filter_channel);
        CL_CHECK_ERRORS(status);
        int group = 1;
        status = clSetKernelArg(kernel, index++, sizeof(int), &group);
        CL_CHECK_ERRORS(status);
      }
    }

    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  }
}

void DWConvAddBnRelu(framework::CLHelper *cl_helper,
                     const ConvParam<GPU_CL> &param, bool ifRelu,
                     const framework::CLImage *biase,
                     const framework::CLImage *new_scale,
                     const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];
  int w_blk_size = 2;
  int w_blk = (w + w_blk_size - 1) / w_blk_size;

  default_work_size[1] = w_blk;
  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();

  auto output = param.Output()->GetCLImage();
  int stride = param.Strides()[0];
  int pad = param.Paddings()[0];
  int dilation = param.Dilations()[0];

  int input_channel = param.Input()->dims()[1];
  int input_height = param.Input()->dims()[2];
  int input_width = param.Input()->dims()[3];

  int output_height = param.Output()->dims()[2];
  int output_width = param.Output()->dims()[3];

  //  DLOG << " w " << w;
  //  DLOG << " nh " << nh;
  //  DLOG << " stride " << stride;
  //  DLOG << " dilation " << dilation;
  //  DLOG << " input width " << input_width;
  //  DLOG << " input height " << input_height;
  //  DLOG << " output width " << output_width;
  //  DLOG << " output height " << output_height;
  //  DLOG << " input dim " << param.Input()->dims();
  //  DLOG << " output dim " << param.Output()->dims();
  //  DLOG << " filter dim " << param.Filter()->dims();

  cl_int status;
  int index = 0;

  status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &w_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &nh);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  if (biase) {
    auto bias_mem = biase->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
    CL_CHECK_ERRORS(status);
  }

  if (new_scale && new_bias) {
    auto new_scale_mem = new_scale->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
    CL_CHECK_ERRORS(status);

    auto new_bias_mem = new_bias->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
    CL_CHECK_ERRORS(status);
  }

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &pad);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_channel);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  auto kernel_work_size = cl_helper->KernelWorkSize(kernel);
  auto tmp0 = default_work_size.data()[0];
  auto tmp1 = default_work_size.data()[1];
  auto tmp2 = default_work_size.data()[2];
  int max_work_size = static_cast<const uint32_t>(kernel_work_size);
  if (preferred_lws_divisor > 1) {
    max_work_size /= preferred_lws_divisor;
  }
  if (preferred_lws > 0 && preferred_lws <= max_work_size) {
    max_work_size = preferred_lws;
  }
  while (tmp1 > max_work_size && max_work_size > 0) {
    tmp1 = tmp1 % 2 == 0 ? tmp1 / 2 : 1;
  }
  while (tmp2 * tmp1 > max_work_size && max_work_size > 0) {
    tmp2 = tmp2 % 2 == 0 ? tmp2 / 2 : 1;
  }
  while (tmp0 * tmp1 * tmp2 > max_work_size && max_work_size > 0) {
    tmp0 = tmp0 % 2 == 0 ? tmp0 / 2 : 1;
  }
  const size_t local_work_size[3] = {static_cast<const uint32_t>(tmp0),
                                     static_cast<const uint32_t>(tmp1),
                                     static_cast<const uint32_t>(tmp2)};
  if (max_work_size > 0 && use_lws) {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), local_work_size, 0, NULL, NULL);
  } else {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
  }

  CL_CHECK_ERRORS(status);
}

void SWConvAddBnRelu(framework::CLHelper *cl_helper,
                     const ConvParam<GPU_CL> &param, bool ifRelu,
                     const framework::CLImage *biase,
                     const framework::CLImage *new_scale,
                     const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  int w_blk_size = 5;
  int w_blk = (w + w_blk_size - 1) / w_blk_size;
  default_work_size[1] = w_blk;

  int h_blk_size = 1;
  int h_blk = (nh + h_blk_size - 1) / h_blk_size;
  default_work_size[2] = h_blk;

  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();

  auto output = param.Output()->GetCLImage();
  int stride = param.Strides()[0];
  int pad = param.Paddings()[0];
  int dilation = param.Dilations()[0];

  int input_channel = param.Input()->dims()[1];
  int input_height = param.Input()->dims()[2];
  int input_width = param.Input()->dims()[3];
  int output_height = param.Output()->dims()[2];
  int output_width = param.Output()->dims()[3];

  cl_int status;
  int index = 0;

  status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &w_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &h_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  if (biase) {
    auto bias_mem = biase->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
    CL_CHECK_ERRORS(status);
  }

  if (new_scale && new_bias) {
    auto new_scale_mem = new_scale->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
    CL_CHECK_ERRORS(status);

    auto new_bias_mem = new_bias->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
    CL_CHECK_ERRORS(status);
  }

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &pad);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_channel);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  auto kernel_work_size = cl_helper->KernelWorkSize(kernel);
  auto tmp0 = default_work_size.data()[0];
  auto tmp1 = default_work_size.data()[1];
  auto tmp2 = default_work_size.data()[2];
  int max_work_size = static_cast<const uint32_t>(kernel_work_size);
  if (preferred_lws_divisor > 1) {
    max_work_size /= preferred_lws_divisor;
  }
  if (preferred_lws > 0 && preferred_lws <= max_work_size) {
    max_work_size = preferred_lws;
  }
  while (tmp1 > max_work_size && max_work_size > 0) {
    tmp1 = tmp1 % 2 == 0 ? tmp1 / 2 : 1;
  }
  while (tmp2 * tmp1 > max_work_size && max_work_size > 0) {
    tmp2 = tmp2 % 2 == 0 ? tmp2 / 2 : 1;
  }
  while (tmp0 * tmp1 * tmp2 > max_work_size && max_work_size > 0) {
    tmp0 = tmp0 % 2 == 0 ? tmp0 / 2 : 1;
  }
  const size_t local_work_size[3] = {static_cast<const uint32_t>(tmp0),
                                     static_cast<const uint32_t>(tmp1),
                                     static_cast<const uint32_t>(tmp2)};
  if (max_work_size > 0 && use_lws) {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), local_work_size, 0, NULL, NULL);
  } else {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
  }
  CL_CHECK_ERRORS(status);
}

void DWConvTransposeAddBnRelu(framework::CLHelper *cl_helper,
                              const ConvTransposeParam<GPU_CL> &param,
                              bool ifRelu, const framework::CLImage *biase,
                              const framework::CLImage *new_scale,
                              const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  int w_blk_size = 1;
  int w_blk = (w + w_blk_size - 1) / w_blk_size;
  default_work_size[1] = w_blk;

  int h_blk_size = 1;
  int h_blk = (nh + h_blk_size - 1) / h_blk_size;
  default_work_size[2] = h_blk;

  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();

  auto output = param.Output()->GetCLImage();
  int stride = param.Strides()[0];
  int pad = param.Paddings()[0];
  int dilation = param.Dilations()[0];

  int input_channel = param.Input()->dims()[1];
  int input_height = param.Input()->dims()[2];
  int input_width = param.Input()->dims()[3];

  int output_height = param.Output()->dims()[2];
  int output_width = param.Output()->dims()[3];

  int filter_height = param.Filter()->dims()[2];
  int filter_width = param.Filter()->dims()[3];

  cl_int status;
  int index = 0;

  status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &w_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &h_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  if (biase) {
    auto bias_mem = biase->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
    CL_CHECK_ERRORS(status);
  }

  if (new_scale && new_bias) {
    auto new_scale_mem = new_scale->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
    CL_CHECK_ERRORS(status);

    auto new_bias_mem = new_bias->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
    CL_CHECK_ERRORS(status);
  }

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &pad);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_channel);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &filter_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &filter_height);
  CL_CHECK_ERRORS(status);

  if (default_work_size.data()[1] % 60 == 0 && use_lws) {
    const size_t local_work_size[3] = {static_cast<const uint32_t>(1),
                                       static_cast<const uint32_t>(60),
                                       static_cast<const uint32_t>(1)};
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), local_work_size, 0, NULL, NULL);
  } else {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
  }
  CL_CHECK_ERRORS(status);
}

void ConvTransposeAddBnRelu_b(framework::CLHelper *cl_helper,
                              const ConvTransposeParam<GPU_CL> &param,
                              bool ifRelu, const framework::CLImage *biase,
                              const framework::CLImage *new_scale,
                              const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  const auto *input = param.Input();
  auto *output = param.Output();
  auto *filter = param.Filter();
  const int n = input->dims()[0];
  const int input_c = input->dims()[1];
  const int input_c_block = (input_c + 3) / 4;
  const int input_width = input->dims()[3];
  const int input_height = input->dims()[2];
  const int output_c = output->dims()[1];
  const int output_c_block = (output_c + 3) / 4;
  const int output_width = output->dims()[3];
  const int output_height = output->dims()[2];

  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();
  auto filterImage = filter->GetCLImage();

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(int), &input_c_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &filterImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);

  const size_t work_size[3] = {(size_t)output_c_block, (size_t)input_width,
                               (size_t)(n * input_height)};

  DLOG << "conv transpose " << input_c_block << input_width << input_height
       << output_width << output_height << work_size[0] << work_size[1]
       << work_size[2];

  clEnqueueNDRangeKernel(cl_helper->CLCommandQueue(), kernel, 3, NULL,
                         work_size, NULL, 0, NULL, NULL);
}
void ConvTransposeAddBnRelu(framework::CLHelper *cl_helper,
                            const ConvTransposeParam<GPU_CL> &param,
                            bool ifRelu, const framework::CLImage *biase,
                            const framework::CLImage *new_scale,
                            const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  int w_blk_size = 1;
  int w_blk = (w + w_blk_size - 1) / w_blk_size;
  default_work_size[1] = w_blk;

  int h_blk_size = 1;
  int h_blk = (nh + h_blk_size - 1) / h_blk_size;
  default_work_size[2] = h_blk;

  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();

  auto output = param.Output()->GetCLImage();
  int stride = param.Strides()[0];
  int pad = param.Paddings()[0];
  int dilation = param.Dilations()[0];

  int input_channel = param.Input()->dims()[1];
  int input_height = param.Input()->dims()[2];
  int input_width = param.Input()->dims()[3];

  int output_height = param.Output()->dims()[2];
  int output_width = param.Output()->dims()[3];

  int filter_height = param.Filter()->dims()[2];
  int filter_width = param.Filter()->dims()[3];

  cl_int status;
  int index = 0;

  status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &w_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &h_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  if (biase) {
    auto bias_mem = biase->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
    CL_CHECK_ERRORS(status);
  }

  if (new_scale && new_bias) {
    auto new_scale_mem = new_scale->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
    CL_CHECK_ERRORS(status);

    auto new_bias_mem = new_bias->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
    CL_CHECK_ERRORS(status);
  }

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &pad);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_channel);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &filter_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &filter_height);
  CL_CHECK_ERRORS(status);

  if (default_work_size.data()[1] % 60 == 0 && use_lws) {
    const size_t local_work_size[3] = {static_cast<const uint32_t>(1),
                                       static_cast<const uint32_t>(60),
                                       static_cast<const uint32_t>(1)};
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), local_work_size, 0, NULL, NULL);
  } else {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
  }
  CL_CHECK_ERRORS(status);
}
void ConvTranspose3x3s2AddBnRelu(framework::CLHelper *cl_helper,
                                 const ConvTransposeParam<GPU_CL> &param,
                                 bool ifRelu, const framework::CLImage *biase,
                                 const framework::CLImage *new_scale,
                                 const framework::CLImage *new_bias) {
  auto kernel = cl_helper->KernelAt(0);
  auto default_work_size = cl_helper->DefaultWorkSize(*param.Output());
  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  int w_blk_size = 5;
  int w_blk = (w + w_blk_size - 1 + 5) / w_blk_size / 2 * 2;
  default_work_size[1] = w_blk;

  int h_blk_size = 1;
  int h_blk = (nh + h_blk_size - 1) / h_blk_size;
  default_work_size[2] = h_blk;

  auto input = param.Input()->GetCLImage();
  auto filter = param.Filter()->GetCLImage();

  auto output = param.Output()->GetCLImage();
  int stride = param.Strides()[0];
  int pad = param.Paddings()[0];
  int dilation = param.Dilations()[0];

  int input_channel = param.Input()->dims()[1];
  int input_height = param.Input()->dims()[2];
  int input_width = param.Input()->dims()[3];

  int output_height = param.Output()->dims()[2];
  int output_width = param.Output()->dims()[3];

  int filter_height = param.Filter()->dims()[2];
  int filter_width = param.Filter()->dims()[3];

  cl_int status;
  int index = 0;

  status = clSetKernelArg(kernel, index++, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &w_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &h_blk);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &input);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &filter);
  CL_CHECK_ERRORS(status);

  if (biase) {
    auto bias_mem = biase->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &bias_mem);
    CL_CHECK_ERRORS(status);
  }

  if (new_scale && new_bias) {
    auto new_scale_mem = new_scale->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_scale_mem);
    CL_CHECK_ERRORS(status);

    auto new_bias_mem = new_bias->GetCLImage();
    status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &new_bias_mem);
    CL_CHECK_ERRORS(status);
  }

  status = clSetKernelArg(kernel, index++, sizeof(cl_mem), &output);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &pad);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, index++, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_channel);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &filter_width);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, index++, sizeof(int), &filter_height);
  CL_CHECK_ERRORS(status);

  auto kernel_work_size = cl_helper->KernelWorkSize(kernel);
  auto tmp0 = default_work_size.data()[0];
  auto tmp1 = default_work_size.data()[1];
  auto tmp2 = default_work_size.data()[2];
  int max_work_size = static_cast<const uint32_t>(kernel_work_size);
  if (preferred_lws_divisor > 1) {
    max_work_size /= preferred_lws_divisor;
  }
  if (preferred_lws > 0 && preferred_lws <= max_work_size) {
    max_work_size = preferred_lws;
  }
  while (tmp1 > max_work_size && max_work_size > 0) {
    tmp1 = tmp1 % 2 == 0 ? tmp1 / 2 : 1;
  }
  while (tmp2 * tmp1 > max_work_size && max_work_size > 0) {
    tmp2 = tmp2 % 2 == 0 ? tmp2 / 2 : 1;
  }
  while (tmp0 * tmp1 * tmp2 > max_work_size && max_work_size > 0) {
    tmp0 = tmp0 % 2 == 0 ? tmp0 / 2 : 1;
  }
  const size_t local_work_size[3] = {static_cast<const uint32_t>(tmp0),
                                     static_cast<const uint32_t>(tmp1),
                                     static_cast<const uint32_t>(tmp2)};
  if (max_work_size > 0 && use_lws) {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), local_work_size, 0, NULL, NULL);
  } else {
    status = clEnqueueNDRangeKernel(
        cl_helper->CLCommandQueue(), kernel, default_work_size.size(), NULL,
        default_work_size.data(), NULL, 0, NULL, NULL);
  }
  CL_CHECK_ERRORS(status);
}
}  // namespace operators
}  // namespace paddle_mobile
