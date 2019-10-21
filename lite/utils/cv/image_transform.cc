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
#include "lite/utils/cv/image_transform.h"
#include <arm_neon.h>
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
void ImageTransform::resize(const uint8_t* src,
                            uint8_t* dst,
                            ImageFormat srcFormat,
                            int srcw,
                            int srch,
                            int dstw,
                            int dsth) {}
void ImageTransform::rotate(const uint8_t* src,
                            uint8_t* dst,
                            ImageFormat srcFormat,
                            int srcw,
                            int srch,
                            float degree) {}
void ImageTransform::flip(const uint8_t* src,
                          uint8_t* dst,
                          ImageFormat srcFormat,
                          int srcw,
                          int srch,
                          FlipParm flip_param) {}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
