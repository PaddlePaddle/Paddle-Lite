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

#ifdef CROP_OP

#include "operators/kernel/crop_kernel.h"
#include "fpga/KD/pes/crop_pe.hpp"

namespace paddle_mobile {
namespace operators {

template <>
bool CropKernel<FPGA, float>::Init(CropParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  zynqmp::CropPE& pe = param->context().pe<zynqmp::CropPE>();
  zynqmp::CropParam& crop_param = pe.param();
  crop_param.input = param->InputX()->zynqmpTensor();
  crop_param.output = param->Out()->zynqmpTensor();
  crop_param.offsets = param->Offsets();
  crop_param.axis = param->Axis();
  crop_param.shape = param->Shape();
  // crop_param.relu.enabled = false;

  pe.init();
  pe.apply();
  return true;
}

template <>
void CropKernel<FPGA, float>::Compute(const CropParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::CropPE& pe = context.pe<zynqmp::CropPE>();
  pe.dispatch();

  // param.Out()->zynqmpTensor()->saveToFile("crop_", true);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
