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
#include "lite/core/tensor.h"
#include "lite/utils/cv/cv_enum.h"
#include "lite/utils/cv/image2tensor.h"
#include "lite/utils/cv/image_convert.h"
#include "lite/utils/cv/image_flip.h"
#include "lite/utils/cv/image_rotate.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::LayOut LayOut;
#define PI 3.14159265f
#define Degrees2Radians(degrees) ((degrees) * (SK_ScalarPI / 180))
#define Radians2Degrees(radians) ((radians) * (180 / SK_ScalarPI))
#define ScalarNearlyZero (1.0f / (1 << 12))

typedef struct {
  int ih;
  int iw;
  int oh;
  int ow;
  FlipParam flip_param;
  float rotate_param;
} TransParam;

class ImagePreprocess {
 public:
  // init
  ImagePreprocess(ImageFormat srcFormat,
                  ImageFormat dstFormat,
                  TransParam param);
  // 颜色空间转换
  void imageCovert(const uint8_t* src, uint8_t* dst);
  void imageCovert(const uint8_t* src,
                   uint8_t* dst,
                   ImageFormat srcFormat,
                   ImageFormat dstFormat);
  // 图像resize
  void imageResize(const uint8_t* src, uint8_t* dst);
  void imageResize(const uint8_t* src,
                   uint8_t* dst,
                   ImageFormat srcFormat,
                   int srcw,
                   int srch,
                   int dstw,
                   int dsth);
  // 图像转换
  void imageRotate(const uint8_t* src, uint8_t* dst);
  void imageRotate(const uint8_t* src,
                   uint8_t* dst,
                   ImageFormat srcFormat,
                   int srcw,
                   int srch,
                   float degree);
  void imageFlip(const uint8_t* src, uint8_t* dst);
  void imageFlip(const uint8_t* src,
                 uint8_t* dst,
                 ImageFormat srcFormat,
                 int srcw,
                 int srch,
                 FlipParam flip_param);
  // image2Tensor and normalize
  void image2Tensor(const uint8_t* src,
                    Tensor* dstTensor,
                    LayOut layout,
                    float* means,
                    float* scales);

 private:
  ImageFormat srcFormat_;
  ImageFormat dstFormat_;
  TransParam transParam_;
};
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
