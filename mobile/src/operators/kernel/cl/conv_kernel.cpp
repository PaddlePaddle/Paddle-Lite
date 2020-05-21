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
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

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

  const std::string conv_kernel_file = "conv_kernel.cl";
  const std::string wino_kernel_file = "winograd_transform.cl";

  if (param->Filter()->dims()[2] == 1 && param->Filter()->dims()[3] == 1) {
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW1x1_FLOAT;
    param->Filter()->InitNImage(cl_helper_.CLContext(),
                                cl_helper_.CLCommandQueue());

    if (param->Input()->dims()[1] % 4 == 0) {
      this->cl_helper_.AddKernel("conv_1x1_simple", conv_kernel_file);
    } else {
      this->cl_helper_.AddKernel("conv_1x1_wrapped", conv_kernel_file);
    }
    DLOG << "conv 1x1";

  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] == 3) {
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    if (param->Strides()[0] == 1 && param->Dilations()[0] == 1) {
      param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3S1_FLOAT;
      this->cl_helper_.AddKernel("depth_conv_3x3s1", conv_kernel_file);
    } else {
      param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3_FLOAT;
      this->cl_helper_.AddKernel("depth_conv_3x3", conv_kernel_file);
    }
    DLOG << "depth_conv 3x3";

  } else if (param->Filter()->dims()[1] == 1 &&
             param->Input()->dims()[1] == param->Output()->dims()[1] &&
             param->Filter()->dims()[2] != 3) {
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    param->ExecMode() = ConvParam<GPU_CL>::EXEC_DEPTHWISEBASIC_FLOAT;
    this->cl_helper_.AddKernel("depth_conv", conv_kernel_file);
  } else if (param->Filter()->dims()[2] == 3 &&
             param->Filter()->dims()[3] == 3) {
    //    if (param->Strides()[0] == param->Strides()[1] &&
    //        param->Strides()[0] == 1 && param->Input()->dims()[2] >= 32) {
    //      param->ExecMode() = ConvParam<GPU_CL>::EXEC_WINOGRAD3X3_FLOAT;
    //      this->cl_helper_.AddKernel("winograd_filter_transform_2x2",
    //                                 wino_kernel_file);
    //      this->cl_helper_.AddKernel("winograd_input_transform_2x2",
    //                                 wino_kernel_file);
    //      this->cl_helper_.AddKernel("matmul", "matmul.cl");
    //      this->cl_helper_.AddKernel("winograd_output_transform_2x2",
    //                                 wino_kernel_file);
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
      this->cl_helper_.AddKernel("conv_3x3", conv_kernel_file);
    } else {
      param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_FLOAT;
      this->cl_helper_.AddKernel("conv_3x3spl", conv_kernel_file);
    }

    //    }
    DLOG << "conv 3x3";
  } else if (param->Filter()->dims()[2] == 7 &&
             param->Filter()->dims()[3] == 7) {
    param->ExecMode() = ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW7x7_FLOAT;
    param->Filter()->InitCLImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());

    this->cl_helper_.AddKernel("conv_7x7", conv_kernel_file);
    //    }
    DLOG << "conv 7x7";
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }

  return true;
}

template <>
void ConvKernel<GPU_CL, float>::Compute(const ConvParam<GPU_CL> &param) {
  switch (param.ExecMode()) {
    case ConvParam<GPU_CL>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<4, 3>(&this->cl_helper_, param);
      break;
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW1x1_FLOAT:
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_WITH_GROUP_FLOAT:
    case ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3_FLOAT:
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW7x7_FLOAT:
    case ConvParam<GPU_CL>::EXEC_DEPTHWISEBASIC_FLOAT:
      ConvAddBnRelu(&this->cl_helper_, param);
      break;
    case ConvParam<GPU_CL>::EXEC_DEPTHWISE3x3S1_FLOAT:
      DWConvAddBnRelu(&this->cl_helper_, param);
      break;
    case ConvParam<GPU_CL>::EXEC_SLIDINGWINDOW3x3_FLOAT:
      SWConvAddBnRelu(&this->cl_helper_, param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
}

template class ConvKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
