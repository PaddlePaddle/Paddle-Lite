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
#ifdef LITE_WITH_FPGA
#include "lite/utils/cv/image2tensor_fpga.h"
#endif

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
__attribute__((visibility("default"))) void ImagePreprocess::image_convert(
    const uint8_t* src, uint8_t* dst) {
  ImageConvert img_convert;
  img_convert.choose(src,
                     dst,
                     this->srcFormat_,
                     this->dstFormat_,
                     this->transParam_.iw,
                     this->transParam_.ih);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_convert(
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

__attribute__((visibility("default"))) void ImagePreprocess::image_convert(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    ImageFormat dstFormat,
    int srcw,
    int srch) {
  ImageConvert img_convert;
  img_convert.choose(src, dst, srcFormat, dstFormat, srcw, srch);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_resize(
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

__attribute__((visibility("default"))) void ImagePreprocess::image_resize(
    const uint8_t* src, uint8_t* dst) {
  int srcw = this->transParam_.iw;
  int srch = this->transParam_.ih;
  int dstw = this->transParam_.ow;
  int dsth = this->transParam_.oh;
  auto srcFormat = this->dstFormat_;
  ImageResize img_resize;
  img_resize.choose(src, dst, srcFormat, srcw, srch, dstw, dsth);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_rotate(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    float degree) {
  ImageRotate img_rotate;
  img_rotate.choose(src, dst, srcFormat, srcw, srch, degree);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_rotate(
    const uint8_t* src, uint8_t* dst) {
  auto srcw = this->transParam_.ow;
  auto srch = this->transParam_.oh;
  auto srcFormat = this->dstFormat_;
  auto degree = this->transParam_.rotate_param;
  ImageRotate img_rotate;
  img_rotate.choose(src, dst, srcFormat, srcw, srch, degree);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_flip(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    FlipParam flip_param) {
  ImageFlip img_flip;
  img_flip.choose(src, dst, srcFormat, srcw, srch, flip_param);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_flip(
    const uint8_t* src, uint8_t* dst) {
  auto srcw = this->transParam_.ow;
  auto srch = this->transParam_.oh;
  auto srcFormat = this->dstFormat_;
  auto flip_param = this->transParam_.flip_param;
  ImageFlip img_flip;
  img_flip.choose(src, dst, srcFormat, srcw, srch, flip_param);
}

__attribute__((visibility("default"))) void ImagePreprocess::image_to_tensor(
    const uint8_t* src,
    Tensor* dstTensor,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    LayoutType layout,
    float* means,
    float* scales) {
#ifdef LITE_WITH_FPGA
  if (this->transParam_.ih > 1080) {
    printf("input image height(%d > 1080) is not supported! \n",
           this->transParam_.ih);
    return;
  }
  Image2TensorFpga img2tensor;
  img2tensor.choose(src,
                    dstTensor,
                    this->srcFormat_,
                    this->dstFormat_,
                    layout,
                    this->transParam_.iw,
                    this->transParam_.ih,
                    this->transParam_.ow,
                    this->transParam_.oh,
                    means,
                    scales);
#else
  Image2Tensor img2tensor;
  img2tensor.choose(
      src, dstTensor, srcFormat, layout, srcw, srch, means, scales);
#endif
}

__attribute__((visibility("default"))) void ImagePreprocess::image_to_tensor(
    const uint8_t* src,
    Tensor* dstTensor,
    LayoutType layout,
    float* means,
    float* scales) {
#ifdef LITE_WITH_FPGA
  if (this->transParam_.ih > 1080) {
    printf("input image height(%d > 1080) is not supported! \n",
           this->transParam_.ih);
    return;
  }

  Image2TensorFpga img2tensor;
  img2tensor.choose(src,
                    dstTensor,
                    this->srcFormat_,
                    this->dstFormat_,
                    layout,
                    this->transParam_.iw,
                    this->transParam_.ih,
                    this->transParam_.ow,
                    this->transParam_.oh,
                    means,
                    scales);
#else
  Image2Tensor img2tensor;
  img2tensor.choose(src,
                    dstTensor,
                    this->dstFormat_,
                    layout,
                    this->transParam_.ow,
                    this->transParam_.oh,
                    means,
                    scales);
#endif
}

__attribute__((visibility("default"))) void ImagePreprocess::image_crop(
    const uint8_t* src,
    uint8_t* dst,
    ImageFormat srcFormat,
    int srcw,
    int srch,
    int left_x,
    int left_y,
    int dstw,
    int dsth) {
  if (dsth > srch || dstw > srcw) {
    printf("output size(%d, %d) must be less than input size(%d, %d) \n",
           dsth,
           dstw,
           srch,
           srcw);
    return;
  }
  if (left_x > srcw || left_x < 0 || left_y > srch || left_y < 0) {
    printf("left point (%d, %d) should be valid \n", left_x, left_y);
    return;
  }
  if (left_x + dstw > srcw || left_y + dsth > srch) {
    printf("left point (%d, %d) and output size(%d, %d) should be valid \n",
           left_x,
           left_y,
           dstw,
           dsth);
    return;
  }
  int stride = 1;
  if (srcFormat == GRAY) {
    stride = 1;
  } else if (srcFormat == BGR || srcFormat == RGB) {
    stride = 3;
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    stride = 4;
  } else {
    printf("this srcFormat: %d does not support! \n", srcFormat);
    return;
  }
  if (dsth == srch && dstw == srcw) {
    memcpy(dst, src, sizeof(uint8_t) * srch * srcw * stride);
    return;
  }
  const uint8_t* in_ptr = src + left_x * srcw * stride + left_y * stride;
  uint8_t* out_ptr = dst;
  for (int row = 0; row < dsth; row++) {
    const uint8_t* din_ptr = in_ptr + row * srcw * stride;
    for (int col = 0; col < dstw * stride; col++) {
      *out_ptr++ = *din_ptr++;
    }
  }
}

}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
