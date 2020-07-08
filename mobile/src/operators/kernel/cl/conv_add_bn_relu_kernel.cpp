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
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

namespace paddle_mobile {
namespace operators {
template <>
bool ConvAddBNReluKernel<GPU_CL, float>::Init(
    FusionConvAddBNReluParam<GPU_CL> *param) {
  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");

  if (!param->Bias()->isInit()) {
    param->Bias()->InitCLImage(cl_helper_.CLContext(),
                               cl_helper_.CLCommandQueue());
  }

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

  const std::string conv_kernel_file = "conv_kernel.cl";
  const std::string wino_kernel_file = "winograd_transform.cl";
  std::string build_options = "-DBATCH_NORM -DRELU";
  if (param->Output()->dims() == param->Bias()->dims()) {
    build_options += " -DBIASE_ELE";
  } else {
    build_options += " -DBIASE_CH";
  }

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
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW1x1_FLOAT;
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());

    if (param->Input()->dims()[1] % 4 == 0) {
      this->cl_helper_.AddKernel("conv_1x1_simple", conv_kernel_file,
                                 build_options);
    } else {
      this->cl_helper_.AddKernel("conv_1x1_wrapped", conv_kernel_file,
                                 build_options);
    }
  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] == 3) {
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    if (param->Strides()[0] == 1 && param->Dilations()[0] == 1) {
      param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3S1_FLOAT;
      this->cl_helper_.AddKernel("depth_conv_3x3s1", conv_kernel_file,
                                 build_options);
    } else {
      param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3_FLOAT;
      this->cl_helper_.AddKernel("depth_conv_3x3", conv_kernel_file,
                                 build_options);
    }

  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] != 3) {
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    // other depthwise not with filter 3x3
    DLOG << "depth_conv basic ";
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISEBASIC_FLOAT;
    this->cl_helper_.AddKernel("depth_conv", conv_kernel_file, build_options);

  } else if (param->Filter()->dims()[2] == 3 &&
             param->Filter()->dims()[3] == 3) {
    //    if (param->Strides()[0] == param->Strides()[1] &&
    //        param->Strides()[0] == 1 && param->Input()->dims()[2] >= 32) {
    //      param->ExecMode() = ConvParam<GPU_CL>::EXEC_WINOGRAD3X3_FLOAT;
    //      this->cl_helper_.AddKernel("winograd_filter_transform_2x2",
    //                                 wino_kernel_file, build_options);
    //      this->cl_helper_.AddKernel("winograd_input_transform_2x2",
    //                                 wino_kernel_file, build_options);
    //      this->cl_helper_.AddKernel("matmul", "matmul.cl");
    //      this->cl_helper_.AddKernel("winograd_output_transform_2x2",
    //                                 wino_kernel_file, build_options);
    //
    //      winograd_transform_weight<4, 3>(&this->cl_helper_, param->Filter());
    //
    //    } else {
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    // std::cout << " input dim " << param->Input()->dims()[0] << "  "
    //           << param->Input()->dims()[1] << "  " <<
    //           param->Input()->dims()[2]
    //           << "  " << param->Input()->dims()[3] << "  " << std::endl;
    // std::cout << " output dim " << param->Output()->dims()[0] << " "
    //           << param->Output()->dims()[1] << " " <<
    //           param->Output()->dims()[2]
    //           << " " << param->Output()->dims()[3] << " " << std::endl;
    // std::cout << " filter dim " << param->Filter()->dims()[0] << " "
    //           << param->Filter()->dims()[1] << " " <<
    //           param->Filter()->dims()[2]
    //           << " " << param->Filter()->dims()[3] << " " << std::endl;

    if (param->groups > 1) {
      param->ExecMode() =
          ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_WITH_GROUP_FLOAT;
      this->cl_helper_.AddKernel("conv_3x3", conv_kernel_file, build_options);
    } else {
      param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_FLOAT;
      this->cl_helper_.AddKernel("conv_3x3spl", conv_kernel_file,
                                 build_options);
    }
    //    }
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }

  return true;
}

template <>
void ConvAddBNReluKernel<GPU_CL, float>::Compute(
    const FusionConvAddBNReluParam<GPU_CL> &param) {
  switch (param.ExecMode()) {
    case ConvParam<GPU_CL>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<4, 3>(&this->cl_helper_, param, true, param.Bias(),
                            param.NewScale(), param.NewBias());
      break;
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW1x1_FLOAT:
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_WITH_GROUP_FLOAT:
    case ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3_FLOAT:
    case ConvParam<GPU_CL>::EXEC_DEPTHWISEBASIC_FLOAT:
      ConvAddBnRelu(&this->cl_helper_, param, true, param.Bias(),
                    param.NewScale(), param.NewBias());
      break;
    case ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3S1_FLOAT:
      DWConvAddBnRelu(&this->cl_helper_, param, true, param.Bias(),
                      param.NewScale(), param.NewBias());
      break;
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_FLOAT:
      SWConvAddBnRelu(&this->cl_helper_, param, true, param.Bias(),
                      param.NewScale(), param.NewBias());
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
}

template class ConvAddBNReluKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
