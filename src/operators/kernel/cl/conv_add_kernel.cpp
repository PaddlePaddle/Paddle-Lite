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
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

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
	ConvAddBnRelu(this->cl_helper_, param, false, param.Bias());
}

template class ConvAddKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
