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

#include "lite/utils/cv/image_transform.h"
#include <arm_neon.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include "lite/utils/cv/image_flip.h"
#include "lite/utils/cv/image_rotate.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
void ImageTransform::rotate(const uint8_t* src,
                            uint8_t* dst,
                            ImageFormat srcFormat,
                            int srcw,
                            int srch,
                            float degree) {
  if (srcFormat == GRAY) {
    rotate_hwc1(src, dst, srcw, srch, degree);
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    rotate_hwc2(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    rotate_hwc3(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    rotate_hwc4(src, dst, srcw, srch, degree);
  }
}
void ImageTransform::flip(const uint8_t* src,
                          uint8_t* dst,
                          ImageFormat srcFormat,
                          int srcw,
                          int srch,
                          FlipParam flip_param) {
  if (srcFormat == GRAY) {
    flip_hwc1(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    flip_hwc2(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    flip_hwc3(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    flip_hwc4(src, dst, srcw, srch, flip_param);
  }
}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
