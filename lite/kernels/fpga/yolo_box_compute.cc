// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/fpga/yolo_box_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void YoloBoxCompute::PrepareForRun() {
  auto& param = Param<operators::YoloBoxParam>();
  lite::Tensor* X = param.X;
  lite::Tensor* ImgSize = param.ImgSize;
  lite::Tensor* Boxes = param.Boxes;
  lite::Tensor* Scores = param.Scores;

  Boxes->mutable_data<float>();
  Scores->mutable_data<float>();

  zynqmp::YoloBoxParam& yolobox_param = pe_.param();
  yolobox_param.input = X->ZynqTensor();
  yolobox_param.imgSize = ImgSize->ZynqTensor();
  yolobox_param.outputBoxes = Boxes->ZynqTensor();
  yolobox_param.outputScores = Scores->ZynqTensor();
  yolobox_param.downsampleRatio = param.downsample_ratio;
  yolobox_param.anchors = param.anchors;
  yolobox_param.classNum = param.class_num;
  yolobox_param.confThresh = param.conf_thresh;

  pe_.init();
  pe_.apply();
}

void YoloBoxCompute::Run() { pe_.dispatch(); }

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::YoloBoxCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(yolo_box,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::YoloBoxCompute,
                     def_float_size)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
