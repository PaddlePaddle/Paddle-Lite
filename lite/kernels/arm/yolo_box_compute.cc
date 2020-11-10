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

#include "lite/kernels/arm/yolo_box_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void YoloBoxCompute::Run() {
  auto& param = Param<operators::YoloBoxParam>();
  lite::Tensor* X = param.X;
  lite::Tensor* ImgSize = param.ImgSize;
  lite::Tensor* Boxes = param.Boxes;
  lite::Tensor* Scores = param.Scores;
  std::vector<int> anchors = param.anchors;
  int class_num = param.class_num;
  float conf_thresh = param.conf_thresh;
  int downsample_ratio = param.downsample_ratio;
  bool clip_bbox = param.clip_bbox;
  float scale_x_y = param.scale_x_y;
  float bias = -0.5 * (scale_x_y - 1.);
  Boxes->clear();
  Scores->clear();
  lite::arm::math::yolobox(X,
                           ImgSize,
                           Boxes,
                           Scores,
                           anchors,
                           class_num,
                           conf_thresh,
                           downsample_ratio,
                           clip_bbox,
                           scale_x_y,
                           bias);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::YoloBoxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
