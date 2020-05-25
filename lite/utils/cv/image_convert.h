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
#include <stdio.h>
#include "lite/utils/cv/paddle_image_preprocess.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
typedef void (*convert_func)(const uint8_t* src,
                             uint8_t* dst,
                             int srcw,
                             int srch);
class ImageConvert {
 public:
  void choose(const uint8_t* src,
              uint8_t* dst,
              ImageFormat srcFormat,
              ImageFormat dstFormat,
              int srcw,
              int srch);

 private:
  convert_func impl_{nullptr};
};
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
