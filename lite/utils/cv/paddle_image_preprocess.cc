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

#include "lite/utils/cv/paddle_image_preprocess.h"
#include <math.h>
#include <string.h>
#include <algorithm>
#include <climits>
#include "lite/utils/cv/image2tensor.h"
#include "lite/utils/cv/image_convert.h"
#include "lite/utils/cv/image_flip.h"
#include "lite/utils/cv/image_resize.h"
#include "lite/utils/cv/image_rotate.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
#define PI 3.14159265f
#define Degrees2Radians(degrees) ((degrees) * (SK_ScalarPI / 180))
#define Radians2Degrees(radians) ((radians) * (180 / SK_ScalarPI))
#define ScalarNearlyZero (1.0f / (1 << 12))
// init
__attribute__((visibility("default")))
ImagePreprocess::ImagePreprocess(ImageFormat srcFormat,
                                 ImageFormat dstFormat,
                                 TransParam param) {
  this->srcFormat_ = srcFormat;
  this->dstFormat_ = dstFormat;
  this->transParam_ = param;
}
__attribute__((visibility("default"))) void ImagePreprocess::imageConvert(
    const uint8_t* src, uint8_t* dst) {
  ImageConvert img_convert;
  img_convert.choose(src,
                     dst,
                     this->srcFormat_,
                     this->dstFormat_,
                     this->transParam_.iw,
                     this->transParam_.ih);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageConvert(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    ImageFormat dstFormat) {
  ImageConvert img_convert;
  img_convert.choose(src,
                     dst,
                     srcFormat,
                     dstFormat,
                     this->transParam_.iw,
                     this->transParam_.ih);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageResize(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    int dstw,
    int dsth) {
  ImageResize img_resize;
  img_resize.choose(src, dst, srcFormat, srcw, srch, dstw, dsth);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageResize(
    const uint8_t* src, uint8_t* dst) {
  int srcw = this->transParam_.iw;
  int srch = this->transParam_.ih;
  int dstw = this->transParam_.ow;
  int dsth = this->transParam_.oh;
  auto srcFormat = this->dstFormat_;
  ImageResize img_resize;
  img_resize.choose(src, dst, srcFormat, srcw, srch, dstw, dsth);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageRotate(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    float degree) {
  ImageRotate img_rotate;
  img_rotate.choose(src, dst, srcFormat, srcw, srch, degree);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageRotate(
    const uint8_t* src, uint8_t* dst) {
  auto srcw = this->transParam_.ow;
  auto srch = this->transParam_.oh;
  auto srcFormat = this->dstFormat_;
  auto degree = this->transParam_.rotate_param;
  ImageRotate img_rotate;
  img_rotate.choose(src, dst, srcFormat, srcw, srch, degree);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageFlip(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    FlipParam flip_param) {
  ImageFlip img_flip;
  img_flip.choose(src, dst, srcFormat, srcw, srch, flip_param);
}

__attribute__((visibility("default"))) void ImagePreprocess::imageFlip(
    const uint8_t* src, uint8_t* dst) {
  auto srcw = this->transParam_.ow;
  auto srch = this->transParam_.oh;
  auto srcFormat = this->dstFormat_;
  auto flip_param = this->transParam_.flip_param;
  ImageFlip img_flip;
  img_flip.choose(src, dst, srcFormat, srcw, srch, flip_param);
}

__attribute__((visibility("default"))) void ImagePreprocess::image2Tensor(
    const uint8_t* src,
    Tensor* dstTensor,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    LayoutType layout,
    float* means,
    float* scales) {
  Image2Tensor img2tensor;
  img2tensor.choose(
      src, dstTensor, srcFormat, layout, srcw, srch, means, scales);
}

__attribute__((visibility("default"))) void ImagePreprocess::image2Tensor(
    const uint8_t* src,
    Tensor* dstTensor,
    LayoutType layout,
    float* means,
    float* scales) {
  Image2Tensor img2tensor;
  img2tensor.choose(src,
                    dstTensor,
                    this->dstFormat_,
                    layout,
                    this->transParam_.ow,
                    this->transParam_.oh,
                    means,
                    scales);
}

}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
