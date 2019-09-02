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
#ifdef CONV_TRANSPOSE_OP

#include "operators/kernel/conv_transpose_kernel.h"
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvTransposeKernel<GPU_CL, float>::Init(
    ConvTransposeParam<GPU_CL>* param) {
  PADDLE_MOBILE_ENFORCE(param->Strides()[0] == param->Strides()[1] &&
                            param->Paddings()[0] == param->Paddings()[1] &&
                            param->Dilations()[0] == param->Dilations()[1] &&
                            param->Dilations()[0] == 1,
                        "need equal");

  if (param->Filter()->dims()[1] == 1 &&
      param->Input()->dims()[1] == param->Output()->dims()[1]) {
    param->ExecMode() = ConvTransposeParam<GPU_CL>::EXEC_DEPTHWISETRANS_FLOAT;
    param->Filter()->InitDWImage(cl_helper_.CLContext(),
                                 cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("depthwise_transpose",
                               "conv_transpose_kernel.cl");
  } else if (param->Filter()->dims()[2] == 3 &&
             param->Filter()->dims()[3] == 3 && param->Strides()[0] == 2) {
    param->ExecMode() = ConvTransposeParam<GPU_CL>::EXEC_CONVTRANS3x3s2_FLOAT;
    param->Filter()->InitConv2dTransposeFilterCLImage(
        cl_helper_.CLContext(), cl_helper_.CLCommandQueue());
    this->cl_helper_.AddKernel("conv_transpose", "conv_transpose_kernel.cl");
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(" not support ");
  }
  return true;
}

template <>
void ConvTransposeKernel<GPU_CL, float>::Compute(
    const ConvTransposeParam<GPU_CL>& param) {
  switch (param.ExecMode()) {
    case ConvTransposeParam<GPU_CL>::EXEC_DEPTHWISETRANS_FLOAT:
      DWConvTransposeAddBnRelu(&this->cl_helper_, param);
      break;
    case ConvTransposeParam<GPU_CL>::EXEC_CONVTRANS3x3s2_FLOAT:
      ConvTransposeAddBnRelu(&this->cl_helper_, param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION(
          "Invalid convolution transpose execute mode %d", param.ExecMode());
  }
}

template class ConvTransposeKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
