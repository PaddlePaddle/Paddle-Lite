/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include "lite/utils/cv/paddle_image_preprocess.h"

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite_api::DataLayoutType LayoutType;

void rotate(const uint8_t* src, uint8_t* dst, int srcw, int srch, int angle);

void bgr_rotate_hwc(const uint8_t* src, uint8_t* dst, int srcw, int srch, int angle);
void bgra_rotate_hwc(const uint8_t* src, uint8_t* dst, int srcw, int srch, int angle);

// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void flip(const uint8_t* src, uint8_t* dst, int srcw, int srch, int flip_num);

// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void bgr_flip_hwc(const uint8_t* src, uint8_t* dst, int srcw, int srch, int flip_num);
// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void bgra_flip_hwc(const uint8_t* src, uint8_t* dst, int srcw, int srch, int flip_num);

// y_w = srcw, y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv21_resize(const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);

void bgr_resize(const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);
void bgra_resize(const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);

// nv21(yvu)  to BGR: stroe hwc dsth * dstw = srch * (srcw) y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv21_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// nv12(yuv)  to BGR:store hwc dsth * dstw = srch * srcw y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv12_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// nv21(yvu)  to BGRA: stroe hwc dsth * dstw = srch * (srcw) y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv21_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch);

//nv12(yuv)  to BGRA:store hwc dsth * dstw = srch * srcw y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv12_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// bgr output.w == width output.h == height/3
void bgr_to_tensor_hcw(const uint8_t* bgr, Tensor& output, int width,
                       int height, float* means, float* scales);

// bgr output.w == width / 3 output.h == height
void bgr_to_tensor_hwc(const uint8_t* bgr, Tensor& output, int width,
                       int height, float* means, float* scales);

// bgra output.w == width / 4 output.h == height
void bgra_to_tensor_hwc(const uint8_t* bgr, Tensor& output, int width,
                        int height, float* means, float* scales);

// yvu   y_w = width, y_h = height uv_w = width uv_h = 1/2 * height
void nv21_to_tensor(const uint8_t* nv21, Tensor& output, int width,
                    int height, float* means, float* scales);

// yuv  y_w = width, y_h = height uv_w = width uv_h = 1/2 * height
void nv12_to_tensor(const uint8_t* nv12, Tensor& output, int width,
                    int height, float* means, float* scales);

void image_basic_convert(const uint8_t* src, uint8_t* basic_dst,
                         ImageFormat srcFormat, ImageFormat dstFormat,
                         int srcw, int srch, int out_size) {
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
                        int srcw, int srch,
                        int dstw, int dsth) {
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
    memcpy(out_data, in_data, sizeof(uint8_t) * size);
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
      LOG(FATAL) << "anakin doesn't support this type: " << (int)srcFormat;
    }
  }
}
void image_basic_flip(const uint8_t* src,
                      uint8_t* dst,
                      ImageFormat srcFormat,
                      int srcw, int srch,
                      int flip_num) {
  if (flip_num == -1) {
    flip_num = 0; //  xy
  } else if (flip_num == 0) {
    flip_num = 1; //  x
  } else if (flip_num == 1) {
    flip_num = -1; //  y
  }
  if (srcFormat == ImageFormat::GRAY) {
    flip(src, dst, srcw, srch, flip_num);
  } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    bgr_flip_hwc(src, dst, srcw, srch, flip_num);
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    bgra_flip_hwc(src, dst, srcw, srch, flip_num);
  } else {
    LOG(FATAL) << "anakin doesn't support this type: " << (int)srcFormat;
  }
}
void image_basic_rotate(const uint8_t* src,
                        uint8_t* dst,
                        ImageFormat srcFormat,
                        int srcw, int srch,
                        float rotate_num) {
  if (srcFormat == ImageFormat::GRAY) {
    rotate(src, dst, srcw, srch, rotate_num);
  } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    bgr_rotate_hwc(src, dst, srcw, srch, rotate_num);
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    bgra_rotate_hwc(src, dst, srcw, srch, rotate_num);
  } else {
    LOG(FATAL) << "anakin doesn't support this type: " << (int)srcFormat;
  }
}

void image_to_tensor_basic(const uint8_t* in_data,
                           Tensor* dst,
                           ImageFormat srcFormat,
                           LayoutType layout,
                           int srcw,
                           int srch,
                           float* means,
                           float* scales) {
  float* output = dst->mutable_data<float>();
  if (layout == LayoutType::kNCHW &&
      (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB)) {
    bgr_to_tensor_chw_basic(in_data, output, srcw, srch, means, scales, 3);
  } else if (layout == LayoutType::kNCHW && (srcFormat == ImageFormat::BGRA ||
                                             srcFormat == ImageFormat::RGBA)) {
    bgra_to_tensor_hwc(src, *dst, srcw, srch, means, scales);
  } else {
    LOG(FATAL) << "anakin doesn't suppoort other layout or format: " << (int)srcFormat;
  }
}
