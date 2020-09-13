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

#include <arm_neon.h>
#include "lite/core/tensor.h"
#include "lite/utils/cv/paddle_image_preprocess.h"

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite_api::DataLayoutType LayoutType;

void rotate(const uint8_t* src, uint8_t* dst, int srcw, int srch, int angle);

void bgr_rotate_hwc(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int angle);

void bgra_rotate_hwc(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int angle);

// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void flip(const uint8_t* src, uint8_t* dst, int srcw, int srch, int flip_num);

// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void bgr_flip_hwc(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int flip_num);
// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void bgra_flip_hwc(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int flip_num);

// y_w = srcw, y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv21_resize(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);

void bgr_resize(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);

void bgra_resize(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth);

// nv21(yvu)  to BGR: stroe hwc dsth * dstw = srch * (srcw) y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv21_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// nv12(yuv)  to BGR:store hwc dsth * dstw = srch * srcw y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv12_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// nv21(yvu)  to BGRA: stroe hwc dsth * dstw = srch * (srcw) y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv21_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// nv12(yuv)  to BGRA:store hwc dsth * dstw = srch * srcw y_w = srcw
// y_h = srch uv_w = srcw uv_h = 1/2 * srch
void nv12_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch);

// bgr output.w == width output.h == height/3
void bgr_to_tensor_hcw(const uint8_t* bgr,
                       Tensor& output,  // NOLINT
                       int width,
                       int height,
                       float* means,
                       float* scales);

// bgr output.w == width / 3 output.h == height
void bgr_to_tensor_hwc(const uint8_t* bgr,
                       Tensor& output,  // NOLINT
                       int width,
                       int height,
                       float* means,
                       float* scales);

// bgra output.w == width / 4 output.h == height
void bgra_to_tensor_hwc(const uint8_t* bgr,
                        Tensor& output,  // NOLINT
                        int width,
                        int height,
                        float* means,
                        float* scales);

// yvu   y_w = width, y_h = height uv_w = width uv_h = 1/2 * height
void nv21_to_tensor(const uint8_t* nv21,
                    Tensor& output,  // NOLINT
                    int width,
                    int height,
                    float* means,
                    float* scales);

// yuv  y_w = width, y_h = height uv_w = width uv_h = 1/2 * height
void nv12_to_tensor(const uint8_t* nv12,
                    Tensor& output,  // NOLINT
                    int width,
                    int height,
                    float* means,
                    float* scales);

// clang-format on
void image_basic_convert(const uint8_t* src,
                         uint8_t* dst,
                         ImageFormat srcFormat,
                         ImageFormat dstFormat,
                         int srcw,
                         int srch,
                         int out_size);

void image_basic_resize(const uint8_t* src,
                        uint8_t* dst,
                        ImageFormat srcFormat,
                        int srcw,
                        int srch,
                        int dstw,
                        int dsth);

void image_basic_flip(const uint8_t* src,
                      uint8_t* dst,
                      ImageFormat srcFormat,
                      int srcw,
                      int srch,
                      int flip_num);

void image_basic_rotate(const uint8_t* src,
                        uint8_t* dst,
                        ImageFormat srcFormat,
                        int srcw,
                        int srch,
                        float rotate_num);

void image_basic_to_tensor(const uint8_t* in_data,
                           Tensor dst,
                           ImageFormat srcFormat,
                           LayoutType layout,
                           int srcw,
                           int srch,
                           float* means,
                           float* scales);
