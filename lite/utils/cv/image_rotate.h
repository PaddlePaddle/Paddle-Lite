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

#include <stdint.h>
#include <vector>
#include "lite/utils/cv/paddle_image_preprocess.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
class ImageRotate {
 public:
  void choose(const uint8_t* src,
              uint8_t* dst,
              ImageFormat srcFormat,
              int srcw,
              int srch,
              float degree);
};
void rotate_hwc1(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree);
void rotate_hwc3(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree);
void rotate_hwc4(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree);
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
