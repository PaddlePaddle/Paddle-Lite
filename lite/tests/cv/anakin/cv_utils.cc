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

#include "lite/tests/cv/anakin/cv_utils.h"

void image_basic_convert(const uint8_t* src,
                         uint8_t* dst,
                         ImageFormat srcFormat,
                         ImageFormat dstFormat,
                         int srcw,
                         int srch,
                         int out_size) {
  if (srcFormat == dstFormat) {
    // copy
    memcpy(dst, src, sizeof(uint8_t) * out_size);
    return;
  } else {
    if (srcFormat == ImageFormat::NV12 &&
        (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB)) {
      nv12_to_bgr(src, dst, srcw, srch);
    } else if (srcFormat == ImageFormat::NV21 &&
               (dstFormat == ImageFormat::BGR ||
                dstFormat == ImageFormat::RGB)) {
      nv21_to_bgr(src, dst, srcw, srch);
    } else if (srcFormat == ImageFormat::NV12 &&
               (dstFormat == ImageFormat::BGRA ||
                dstFormat == ImageFormat::RGBA)) {
      nv12_to_bgra(src, dst, srcw, srch);
    } else if (srcFormat == ImageFormat::NV21 &&
               (dstFormat == ImageFormat::BGRA ||
                dstFormat == ImageFormat::RGBA)) {
      nv21_to_bgra(src, dst, srcw, srch);
    } else {
      printf("bais-anakin srcFormat: %d, dstFormat: %d does not support! \n",
             srcFormat,
             dstFormat);
    }
  }
}

void image_basic_resize(const uint8_t* src,
                        uint8_t* dst,
                        ImageFormat srcFormat,
                        int srcw,
                        int srch,
                        int dstw,
                        int dsth) {
  int size = srcw * srch;
  if (srcw == dstw && srch == dsth) {
    if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
      size = srcw * (static_cast<int>(1.5 * srch));
    } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
      size = 3 * srcw * srch;
    } else if (srcFormat == ImageFormat::BGRA ||
               srcFormat == ImageFormat::RGBA) {
      size = 4 * srcw * srch;
    }
    memcpy(dst, src, sizeof(uint8_t) * size);
    return;
  } else {
    if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
      nv21_resize(src, dst, srcw, srch, dstw, dsth);
    } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
      bgr_resize(src, dst, srcw, srch, dstw, dsth);
    } else if (srcFormat == ImageFormat::BGRA ||
               srcFormat == ImageFormat::RGBA) {
      bgra_resize(src, dst, srcw, srch, dstw, dsth);
    } else {
      printf("anakin doesn't support this type: %d\n",
             static_cast<int>(srcFormat));
    }
  }
}

void image_basic_flip(const uint8_t* src,
                      uint8_t* dst,
                      ImageFormat srcFormat,
                      int srcw,
                      int srch,
                      int flip_num) {
  if (flip_num == -1) {
    flip_num = 0;  // xy
  } else if (flip_num == 0) {
    flip_num = 1;  // x
  } else if (flip_num == 1) {
    flip_num = -1;  // y
  }
  if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    bgr_flip_hwc(src, dst, srcw, srch, flip_num);
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    bgra_flip_hwc(src, dst, srcw, srch, flip_num);
  } else {
    printf("anakin doesn't support this type: %d\n",
           static_cast<int>(srcFormat));
  }
}

void image_basic_rotate(const uint8_t* src,
                        uint8_t* dst,
                        ImageFormat srcFormat,
                        int srcw,
                        int srch,
                        float rotate_num) {
  if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    bgr_rotate_hwc(src, dst, srcw, srch, rotate_num);
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    bgra_rotate_hwc(src, dst, srcw, srch, rotate_num);
  } else {
    printf("anakin doesn't support this type: %d\n",
           static_cast<int>(srcFormat));
  }
}

void image_basic_to_tensor(const uint8_t* in_data,
                           Tensor dst,
                           ImageFormat srcFormat,
                           LayoutType layout,
                           int srcw,
                           int srch,
                           float* means,
                           float* scales) {
  if (layout == LayoutType::kNCHW &&
      (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB)) {
    bgr_to_tensor_hwc(in_data, dst, srcw, srch, means, scales);
  } else if (layout == LayoutType::kNCHW && (srcFormat == ImageFormat::BGRA ||
                                             srcFormat == ImageFormat::RGBA)) {
    bgra_to_tensor_hwc(in_data, dst, srcw, srch, means, scales);
  } else {
    printf("anakin doesn't support this type: %d\n",
           static_cast<int>(srcFormat));
  }
}
