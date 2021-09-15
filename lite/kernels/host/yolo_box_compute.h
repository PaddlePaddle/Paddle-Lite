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

#pragma once
#include <vector>
#include "lite/backends/host/math/yolo_box.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#ifdef ENABLE_ARM_FP16
using float16_t = __fp16;
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, TargetType TType, PrecisionType PType>
class YoloBoxCompute : public KernelLite<TType, PType> {
 public:
  void Run() {
    auto& param = this->template Param<operators::YoloBoxParam>();
    lite::Tensor* X = param.X;
    lite::Tensor* ImgSize = param.ImgSize;
    lite::Tensor* Boxes = param.Boxes;
    lite::Tensor* Scores = param.Scores;
    std::vector<int> anchors = param.anchors;
    int class_num = param.class_num;
    T conf_thresh = static_cast<T>(param.conf_thresh);
    int downsample_ratio = param.downsample_ratio;
    bool clip_bbox = param.clip_bbox;
    T scale_x_y = static_cast<T>(param.scale_x_y);
    T bias = static_cast<T>(-0.5 * (scale_x_y - 1.));
    Boxes->clear();
    Scores->clear();
    lite::host::math::YoloBox<T>(X,
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

  virtual ~YoloBoxCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
