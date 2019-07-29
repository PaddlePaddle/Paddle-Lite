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

#ifdef YOLOBOX_OP

#include "fpga/KD/pes/yolobox_pe.hpp"
#include "operators/kernel/yolo_box_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool YoloBoxKernel<FPGA, float>::Init(YoloBoxParam<FPGA>* param) {
  param->OutputBoxes()->mutable_data<half>();
  param->OutputScores()->mutable_data<half>();

  zynqmp::YoloBoxPE& pe = param->context().pe<zynqmp::YoloBoxPE>();
  zynqmp::YoloBoxParam& yolobox_param = pe.param();
  yolobox_param.input = param->Input()->zynqmpTensor();
  yolobox_param.imgSize = param->ImgSize()->zynqmpTensor();
  yolobox_param.outputBoxes = param->OutputBoxes()->zynqmpTensor();
  yolobox_param.outputScores = param->OutputScores()->zynqmpTensor();
  yolobox_param.downsampleRatio = param->DownsampleRatio();
  yolobox_param.anchors = param->Anchors();
  yolobox_param.classNum = param->ClassNum();
  yolobox_param.confThresh = param->ConfThresh();

  zynqmp::Tensor* tensor = param->ImgSize()->zynqmpTensor();
  // std::cout << "YoloBoxKernel init ImgSize:" << tensor->shape().channel() << std::endl;

  pe.init();
  pe.apply();

  return true;
}

template <>
void YoloBoxKernel<FPGA, float>::Compute(const YoloBoxParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  zynqmp::YoloBoxPE& pe = context.pe<zynqmp::YoloBoxPE>();
  pe.dispatch();

  // param.OutputBoxes()->zynqmpTensor()->saveToFile("yolobox_OutputBoxes", true);
  // param.OutputScores()->zynqmpTensor()->saveToFile("yolobox_OutputScores", true);
}

template class YoloBoxKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
