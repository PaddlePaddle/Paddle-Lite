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
#include "lite/utils/cv/cv_enum.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParm FlipParm;
typedef void (*rotate_func)(const uint8_t* src,
                            uint8_t* dst,
                            int srcw,
                            int srch);
typedef void (*flip_func)(const uint8_t* src, uint8_t* dst, int srcw, int srch);
typedef void (*resize_func)(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);
class ImageTransform {
 public:
  void rotate(const uint8_t* src,
              uint8_t* dst,
              ImageFormat srcFormat,
              int srcw,
              int srch,
              float degree);

  void flip(const uint8_t* src,
            uint8_t* dst,
            ImageFormat srcFormat,
            int srcw,
            int srch,
            FlipParm flip_param);

  void resize(const uint8_t* src,
              uint8_t* dst,
              ImageFormat srcFormat,
              int srcw,
              int srch,
              int dstw,
              int dsth);

 private:
  rotate_func rotate_impl_{nullptr};
  flip_func flip_impl_{nullptr};
  resize_func resize_impl_{nullptr};
};
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
