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

#ifdef FUSION_CONVADDRELU_OP

#include "operators/kernel/conv_add_relu_kernel.h"
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvAddReluKernel<GPU_CL, float>::Init(
    FusionConvAddReluParam<GPU_CL> *param) {
  PADDLE_MOBILE_ENFORCE(
      param->Filter()->dims()[2] == param->Filter()->dims()[3] &&
          param->Paddings()[0] == param->Paddings()[1],
      "need equal");
  param->Bias()->InitCLImage(cl_helper_.CLContext(),
                             this->cl_helper_.CLCommandQueue());

  int offset = static_cast<int>(param->Filter()->dims()[2]) / 2 -
               static_cast<int>(param->Paddings()[1]);
  param->SetOffset(offset);

  const std::string conv_kernel_file = "conv_kernel.cl";
  const std::string wino_kernel_file = "winograd_transform.cl";
  const std::string build_options = "-DBIASE -DRELU";

  if (param->Filter()->dims()[2] == 1 && param->Filter()->dims()[3] == 1) {
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW1x1_FLOAT;
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_1x1_spl", conv_kernel_file, build_options);

  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] == 3) {
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3_FLOAT;
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("depth_conv_3x3", conv_kernel_file,
                               build_options);

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
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_FLOAT;
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_3x3", conv_kernel_file, build_options);
    //    }

  } else if (param->Filter()->dims()[2] == 7 &&
             param->Filter()->dims()[3] == 7) {
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW7x7_FLOAT;
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_7x7", conv_kernel_file, build_options);

  } else if (param->Filter()->dims()[2] == 5 &&
             param->Filter()->dims()[3] == 5) {
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW5x5_FLOAT;
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_5x5", conv_kernel_file, build_options);

  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }

  return true;
}

template <>
void ConvAddReluKernel<GPU_CL, float>::Compute(
    const FusionConvAddReluParam<GPU_CL> &param) {
  switch (param.ExecMode()) {
    case ConvParam<GPU_CL>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<4, 3>(&this->cl_helper_, param, true, param.Bias());
      break;
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW1x1_FLOAT:
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_FLOAT:
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW5x5_FLOAT:
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW7x7_FLOAT:
    case ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3_FLOAT:
      ConvAddBnRelu(&this->cl_helper_, param, true, param.Bias());
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
}

template class ConvAddReluKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
