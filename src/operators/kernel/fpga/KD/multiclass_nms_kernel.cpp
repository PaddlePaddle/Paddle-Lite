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

#ifdef MULTICLASSNMS_OP

#include "operators/kernel/multiclass_nms_kernel.h"
#include "operators/kernel/fpga/KD/multiclass_nms_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool MultiClassNMSKernel<FPGA, float>::Init(MultiClassNMSParam<FPGA> *param) {
  param->Out()->mutable_data<float>();
  param->Out()->zynqmpTensor()->setAligned(false);
  param->Out()->zynqmpTensor()->setDataLocation(zynqmp::CPU);
  return true;
}

template <>
void MultiClassNMSKernel<FPGA, float>::Compute(
    const MultiClassNMSParam<FPGA> &param) {
  param.InputBBoxes()->zynqmpTensor()->syncToCPU();
  param.InputScores()->zynqmpTensor()->syncToCPU();
  MultiClassNMSCompute<float>(param);

  param.Out()->zynqmpTensor()->saveToFile("detection.txt");
}

template class MultiClassNMSKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
